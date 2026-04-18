"""Tests for the Chunk 1 augmentation primitives.

Covers:
  * transpose_tokens — pitch shift correctness, identity on zero, specials
    and durations untouched, out-of-range returns None, decode still works.
  * sliding_windows — songs shorter than / equal to / longer than the
    window size, hop stride, SOS/EOS framing, BAR placement.
"""

from __future__ import annotations

import pytest
from music21 import note, stream

from src.data.augmentation import (
    DEFAULT_HOP_BARS,
    DEFAULT_TRANSPOSITION_RANGE,
    DEFAULT_WINDOW_BARS,
    sliding_windows,
    transpose_tokens,
)
from src.data.tokenizer import decode_part, encode_part
from src.data.vocab import (
    BAR,
    DUR_OFFSET,
    EOS,
    PITCH_OFFSET,
    SOS,
    VOCAB_SIZE,
    duration_to_token,
    is_duration_token,
    is_pitch_token,
    pitch_to_token,
)


def _build_part(elements_per_measure: list[list[note.GeneralNote]]) -> stream.Part:
    part = stream.Part()
    for els in elements_per_measure:
        m = stream.Measure()
        for el in els:
            m.append(el)
        part.append(m)
    return part


# ---------------------------------------------------------------------------
# transpose_tokens
# ---------------------------------------------------------------------------


class TestTranspose:
    def test_zero_shift_is_identity_copy(self):
        part = _build_part([[note.Note("C4", quarterLength=1.0)]])
        toks = encode_part(part)
        out = transpose_tokens(toks, 0)
        assert out == toks
        assert out is not toks  # must be a new list, not an alias

    def test_shift_up_five_semitones(self):
        # C4 (MIDI 60) -> F4 (MIDI 65).
        part = _build_part([[note.Note("C4", quarterLength=1.0)]])
        toks = encode_part(part)
        out = transpose_tokens(toks, 5)
        assert out is not None
        pitch_tokens = [t for t in out if is_pitch_token(t)]
        assert pitch_tokens == [pitch_to_token(65)]

    def test_shift_down_two_semitones(self):
        # C4 (60) -> Bb3 (58).
        part = _build_part([[note.Note("C4", quarterLength=1.0)]])
        toks = encode_part(part)
        out = transpose_tokens(toks, -2)
        assert out is not None
        assert [t for t in out if is_pitch_token(t)] == [pitch_to_token(58)]

    def test_specials_and_durations_untouched(self):
        part = _build_part([
            [note.Note("C4", quarterLength=1.0)],
            [note.Rest(quarterLength=1.0)],
        ])
        toks = encode_part(part)
        out = transpose_tokens(toks, 3)
        assert out is not None
        # Every non-pitch token must survive unchanged at the same index.
        for original, shifted in zip(toks, out):
            if is_pitch_token(original):
                assert shifted == original + 3
            else:
                assert shifted == original

    def test_out_of_range_returns_none(self):
        # C-1 (MIDI 0) shifted down one octave -> MIDI -12, out of range.
        toks = [SOS, pitch_to_token(0), duration_to_token(4), EOS]
        assert transpose_tokens(toks, -12) is None
        # G9 (MIDI 127) shifted up one -> MIDI 128, out of range.
        toks = [SOS, pitch_to_token(127), duration_to_token(4), EOS]
        assert transpose_tokens(toks, 1) is None

    def test_round_trip_survives_transposition(self):
        part = _build_part([
            [note.Note("C4", quarterLength=1.0), note.Note("E4", quarterLength=1.0)],
            [note.Note("G4", quarterLength=2.0)],
        ])
        toks = encode_part(part)
        shifted = transpose_tokens(toks, 7)  # up a perfect fifth
        assert shifted is not None
        decoded = decode_part(shifted)
        midis = [n.pitch.midi for n in decoded.flatten().notes]
        assert midis == [67, 71, 74]  # G4, B4, D5

    def test_full_default_range_size(self):
        assert len(DEFAULT_TRANSPOSITION_RANGE) == 12
        assert 0 in DEFAULT_TRANSPOSITION_RANGE

    @pytest.mark.parametrize("semitones", DEFAULT_TRANSPOSITION_RANGE)
    def test_middle_range_always_fits(self, semitones: int):
        # A middle-register note survives every transposition in the default range.
        toks = [SOS, pitch_to_token(60), duration_to_token(4), EOS]
        assert transpose_tokens(toks, semitones) is not None

    def test_all_output_tokens_in_vocab(self):
        toks = [SOS, pitch_to_token(60), duration_to_token(4), BAR,
                pitch_to_token(67), duration_to_token(8), EOS]
        for s in range(-7, 8):
            out = transpose_tokens(toks, s)
            if out is None:
                continue
            assert all(0 <= t < VOCAB_SIZE for t in out)


# ---------------------------------------------------------------------------
# sliding_windows
# ---------------------------------------------------------------------------


def _make_bar_tokens(n_bars: int) -> list[int]:
    """Build a stream of N bars each containing a single C4 quarter note."""
    tokens: list[int] = [SOS]
    for i in range(n_bars):
        if i > 0:
            tokens.append(BAR)
        tokens.append(pitch_to_token(60 + i))  # distinguish bars by pitch
        tokens.append(duration_to_token(4))
    tokens.append(EOS)
    return tokens


def _count_bars(window: list[int]) -> int:
    # A window with N bars has N-1 BAR tokens between them.
    return window.count(BAR) + 1


class TestSlidingWindows:
    def test_shorter_than_window_yields_one_window(self):
        toks = _make_bar_tokens(3)
        windows = sliding_windows(toks, window_bars=8, hop_bars=4)
        assert len(windows) == 1
        assert windows[0][0] == SOS and windows[0][-1] == EOS
        assert _count_bars(windows[0]) == 3

    def test_exactly_window_size_yields_one_window(self):
        toks = _make_bar_tokens(8)
        windows = sliding_windows(toks, window_bars=8, hop_bars=4)
        assert len(windows) == 1
        assert _count_bars(windows[0]) == 8

    def test_twelve_bars_window_8_hop_4_yields_two_windows(self):
        # Bars 0..7 and 4..11.
        toks = _make_bar_tokens(12)
        windows = sliding_windows(toks, window_bars=8, hop_bars=4)
        assert len(windows) == 2
        assert _count_bars(windows[0]) == 8
        assert _count_bars(windows[1]) == 8

    def test_sixteen_bars_window_8_hop_4_yields_three_windows(self):
        # Bars 0..7, 4..11, 8..15.
        toks = _make_bar_tokens(16)
        windows = sliding_windows(toks, window_bars=8, hop_bars=4)
        assert len(windows) == 3

    def test_window_pitches_match_source_bars(self):
        # Each bar in _make_bar_tokens carries pitch 60 + bar_idx.
        toks = _make_bar_tokens(12)
        windows = sliding_windows(toks, window_bars=8, hop_bars=4)
        # First window covers bars 0..7 -> pitches 60..67.
        first_pitches = [
            t - PITCH_OFFSET for t in windows[0] if is_pitch_token(t)
        ]
        assert first_pitches == list(range(60, 68))
        # Second window covers bars 4..11 -> pitches 64..71.
        second_pitches = [
            t - PITCH_OFFSET for t in windows[1] if is_pitch_token(t)
        ]
        assert second_pitches == list(range(64, 72))

    def test_windows_frame_with_sos_eos(self):
        toks = _make_bar_tokens(16)
        for w in sliding_windows(toks, window_bars=8, hop_bars=4):
            assert w[0] == SOS
            assert w[-1] == EOS
            # No stray SOS/EOS in the interior.
            assert SOS not in w[1:]
            assert EOS not in w[:-1]

    def test_empty_input_returns_no_windows(self):
        assert sliding_windows([SOS, EOS]) == []
        assert sliding_windows([]) == []

    def test_rejects_non_positive_parameters(self):
        toks = _make_bar_tokens(4)
        with pytest.raises(ValueError):
            sliding_windows(toks, window_bars=0, hop_bars=4)
        with pytest.raises(ValueError):
            sliding_windows(toks, window_bars=8, hop_bars=0)

    def test_default_params_match_plan(self):
        # Guard against accidental drift from the Day 2 plan (8/4).
        assert DEFAULT_WINDOW_BARS == 8
        assert DEFAULT_HOP_BARS == 4

    def test_decodes_cleanly_after_windowing(self):
        # The whole point: windows should still be valid token streams.
        toks = _make_bar_tokens(10)
        for w in sliding_windows(toks, window_bars=8, hop_bars=4):
            decoded = decode_part(w)
            assert len(list(decoded.flatten().notes)) > 0
            assert all(is_pitch_token(t) or is_duration_token(t)
                       or t in {SOS, EOS, BAR} for t in w)
