"""Round-trip + grammar tests for the Day 2 tokenizer core.

Covers:
  * ``vocab`` — token-space partition, pitch/duration encode+decode, quantisation.
  * ``tokenizer`` — hand-built Part round-trip, measure/BAR semantics, rest
    handling, chord collapse, orphan-token tolerance on decode.
  * Corpus integration — tokenise one real JSB chorale soprano and confirm
    end-to-end note-count preservation.

Run from repo root::

    pytest
"""

from __future__ import annotations

import logging

import pytest
from music21 import chord, corpus, note, stream

from src.data.load import is_clean_satb
from src.data.tokenizer import decode_part, encode_part
from src.data.vocab import (
    BAR,
    DUR_BUCKETS,
    DUR_OFFSET,
    EOS,
    NUM_PITCHES,
    PAD,
    PITCH_OFFSET,
    REST,
    SOS,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
    duration_to_token,
    is_duration_token,
    is_pitch_token,
    is_special_token,
    pitch_to_token,
    token_to_duration,
    token_to_pitch,
)


# ---------------------------------------------------------------------------
# vocab
# ---------------------------------------------------------------------------


class TestVocab:
    def test_vocab_size_matches_layout(self):
        # 5 specials + 128 pitches + 8 duration buckets = 141
        assert VOCAB_SIZE == 5 + NUM_PITCHES + len(DUR_BUCKETS) == 141

    def test_special_ids_unique_and_below_pitch_offset(self):
        assert len(SPECIAL_TOKENS) == 5
        assert max(SPECIAL_TOKENS) < PITCH_OFFSET

    @pytest.mark.parametrize("midi", [0, 60, 127])
    def test_pitch_round_trip(self, midi: int):
        assert token_to_pitch(pitch_to_token(midi)) == midi

    @pytest.mark.parametrize("bad", [-1, 128, 256])
    def test_pitch_rejects_out_of_range(self, bad: int):
        with pytest.raises(ValueError):
            pitch_to_token(bad)

    @pytest.mark.parametrize("sixteenths", DUR_BUCKETS)
    def test_exact_bucket_is_identity(self, sixteenths: int):
        assert token_to_duration(duration_to_token(sixteenths)) == sixteenths

    def test_duration_snaps_to_nearest_bucket(self):
        # 5 is between buckets 4 and 6; 6 is closer (tie-breaker also favours 6).
        assert token_to_duration(duration_to_token(5)) == 6
        # 7 is between 6 and 8; both distance 1, ties go up -> 8.
        assert token_to_duration(duration_to_token(7)) == 8
        # 0.25 is far below smallest bucket (1); snaps up.
        assert token_to_duration(duration_to_token(0.5)) == 1
        # 100 is way above; snaps down to 16.
        assert token_to_duration(duration_to_token(100)) == 16

    def test_duration_rejects_non_positive(self):
        with pytest.raises(ValueError):
            duration_to_token(0)
        with pytest.raises(ValueError):
            duration_to_token(-1)

    def test_predicates_partition_vocab(self):
        # Every token in [0, VOCAB_SIZE) belongs to exactly one category.
        for tok in range(VOCAB_SIZE):
            hits = (
                int(is_special_token(tok))
                + int(is_pitch_token(tok))
                + int(is_duration_token(tok))
            )
            assert hits == 1, f"token {tok} matches {hits} categories"

    def test_out_of_vocab_tokens_match_nothing(self):
        assert not is_pitch_token(VOCAB_SIZE)
        assert not is_duration_token(VOCAB_SIZE + 10)
        assert not is_special_token(99)  # 99 is a pitch token, not special

    def test_bar_and_rest_ids_in_special_set(self):
        assert BAR in SPECIAL_TOKENS
        assert REST in SPECIAL_TOKENS
        assert PAD in SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# tokenizer — hand-built fixtures (no disk I/O, fully deterministic)
# ---------------------------------------------------------------------------


def _build_part(elements_per_measure: list[list[note.GeneralNote]]) -> stream.Part:
    part = stream.Part()
    for els in elements_per_measure:
        m = stream.Measure()
        for el in els:
            m.append(el)
        part.append(m)
    return part


class TestEncodeGrammar:
    def test_empty_part_yields_only_sos_eos(self):
        assert encode_part(stream.Part()) == [SOS, EOS]

    def test_sos_eos_framing_can_be_disabled(self):
        part = _build_part([[note.Note("C4", quarterLength=1.0)]])
        toks = encode_part(part, include_sos_eos=False)
        assert SOS not in toks and EOS not in toks

    def test_pitch_and_duration_pair_order(self):
        # Single quarter-note C4 in one measure.
        part = _build_part([[note.Note("C4", quarterLength=1.0)]])
        toks = encode_part(part)
        # SOS, PITCH(60), DUR(4=quarter), EOS
        assert toks[0] == SOS
        assert toks[-1] == EOS
        assert token_to_pitch(toks[1]) == 60
        assert token_to_duration(toks[2]) == 4
        assert len(toks) == 4

    def test_bar_between_measures(self):
        part = _build_part([
            [note.Note("C4", quarterLength=1.0)],
            [note.Note("D4", quarterLength=1.0)],
        ])
        toks = encode_part(part)
        # SOS, C4, Q, BAR, D4, Q, EOS
        assert toks == [
            SOS,
            pitch_to_token(60), duration_to_token(4),
            BAR,
            pitch_to_token(62), duration_to_token(4),
            EOS,
        ]

    def test_rest_encodes_as_rest_plus_duration(self):
        part = _build_part([[note.Rest(quarterLength=2.0)]])
        toks = encode_part(part)
        assert toks == [SOS, REST, duration_to_token(8), EOS]

    def test_all_tokens_in_vocab_range(self):
        part = _build_part([[note.Note("C4", quarterLength=q) for q in (0.25, 0.5, 1.0, 2.0, 4.0)]])
        toks = encode_part(part)
        assert all(0 <= t < VOCAB_SIZE for t in toks)

    def test_chord_collapses_to_top_note_with_warning(self, caplog):
        # Chord in a monophonic part is a data defect; we take the top pitch.
        ch = chord.Chord(["C4", "E4", "G4"], quarterLength=1.0)
        part = _build_part([[ch]])
        with caplog.at_level(logging.WARNING, logger="src.data.tokenizer"):
            toks = encode_part(part)
        assert any("chord" in rec.message.lower() for rec in caplog.records)
        # G4 = MIDI 67 (the top note of C major triad in this voicing).
        assert token_to_pitch(toks[1]) == 67

    def test_zero_duration_elements_are_skipped(self):
        grace = note.Note("C4")
        grace.quarterLength = 0.0
        part = _build_part([[grace, note.Note("D4", quarterLength=1.0)]])
        toks = encode_part(part)
        # Grace note gone; only D4 survives.
        assert toks == [SOS, pitch_to_token(62), duration_to_token(4), EOS]


# ---------------------------------------------------------------------------
# tokenizer — decode + round-trip
# ---------------------------------------------------------------------------


class TestDecodeRoundTrip:
    def test_round_trip_preserves_note_count(self):
        part = _build_part([
            [note.Note("C4", quarterLength=1.0), note.Note("D4", quarterLength=1.0)],
            [note.Rest(quarterLength=1.0), note.Note("E4", quarterLength=1.0)],
        ])
        decoded = decode_part(encode_part(part))
        orig_count = len(list(part.flatten().notesAndRests))
        decoded_count = len(list(decoded.flatten().notesAndRests))
        assert orig_count == decoded_count == 4

    def test_round_trip_preserves_pitches_and_durations(self):
        part = _build_part([
            [note.Note("C4", quarterLength=1.0), note.Note("G4", quarterLength=2.0)],
        ])
        decoded = decode_part(encode_part(part))
        elements = list(decoded.flatten().notesAndRests)
        assert [e.pitch.midi for e in elements] == [60, 67]
        # Durations encoded as 16ths {4, 8} -> quarterLength {1.0, 2.0}.
        assert [float(e.quarterLength) for e in elements] == [1.0, 2.0]

    def test_decode_drops_orphan_pitch_with_warning(self, caplog):
        # PITCH token not followed by DUR -> dropped with warning.
        toks = [SOS, pitch_to_token(60), EOS]
        with caplog.at_level(logging.WARNING, logger="src.data.tokenizer"):
            decoded = decode_part(toks)
        assert any("duration" in rec.message.lower() for rec in caplog.records)
        assert len(list(decoded.flatten().notesAndRests)) == 0

    def test_decode_drops_unknown_tokens(self, caplog):
        # VOCAB_SIZE is by definition out of range.
        toks = [SOS, VOCAB_SIZE, pitch_to_token(60), duration_to_token(4), EOS]
        with caplog.at_level(logging.WARNING, logger="src.data.tokenizer"):
            decoded = decode_part(toks)
        assert any("unknown token" in rec.message.lower() for rec in caplog.records)
        # The valid C4 quarter still survives.
        elements = list(decoded.flatten().notesAndRests)
        assert len(elements) == 1 and elements[0].pitch.midi == 60

    def test_decode_creates_separate_measures_at_bar(self):
        part = _build_part([
            [note.Note("C4", quarterLength=1.0)],
            [note.Note("D4", quarterLength=1.0)],
        ])
        decoded = decode_part(encode_part(part))
        measures = list(decoded.getElementsByClass(stream.Measure))
        assert len(measures) == 2
        assert len(list(measures[0].notesAndRests)) == 1
        assert len(list(measures[1].notesAndRests)) == 1


# ---------------------------------------------------------------------------
# Corpus integration — one real JSB chorale end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def jsb_chorale() -> stream.Score:
    for score in corpus.chorales.Iterator():
        if is_clean_satb(score):
            return score
    pytest.skip("no clean SATB chorale available in music21 corpus")


class TestJSBIntegration:
    def test_soprano_round_trip_preserves_note_count(self, jsb_chorale: stream.Score):
        soprano = jsb_chorale.parts[0]
        toks = encode_part(soprano)
        assert all(0 <= t < VOCAB_SIZE for t in toks)
        decoded = decode_part(toks)
        orig = len(list(soprano.flatten().notesAndRests))
        got = len(list(decoded.flatten().notesAndRests))
        assert orig == got, f"note count drift: {orig} -> {got}"

    def test_all_four_voices_round_trip(self, jsb_chorale: stream.Score):
        for part in jsb_chorale.parts:
            toks = encode_part(part)
            assert all(0 <= t < VOCAB_SIZE for t in toks)
            decoded = decode_part(toks)
            assert len(list(decoded.flatten().notesAndRests)) == len(
                list(part.flatten().notesAndRests)
            )
