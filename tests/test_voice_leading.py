"""Tests for the voice-leading post-processor.

Mirrors the synthetic-input patterns in ``tests/test_eval.py`` — build
small hand-crafted token streams, run them through the module, assert
the transformation we expect.
"""

from __future__ import annotations

import logging

from src.data.vocab import (
    BAR,
    DUR_OFFSET,
    PAD,
    REST,
    SATB_RANGES,
    pitch_to_token,
    token_to_pitch,
)
from src.data.vocab import DUR_BUCKETS, EOS, SOS, duration_to_token
from src.postprocess.voice_leading import (
    _align_events_to_grid,
    _tokens_to_events,
    apply_voice_leading,
    coalesce_voice_tokens,
    detect_parallel_motion,
    fit_to_length,
)


VOICES = ("s", "a", "t", "b")
# DUR_OFFSET points at the 16th-note bucket (duration = 1 16th-grid unit),
# which keeps hand-built timelines one step per (pitch, duration) pair.
DUR_16TH = DUR_OFFSET


def _voice_dict(
    s: list[int], a: list[int], t: list[int], b: list[int]
) -> dict[str, list[int]]:
    return {"s": s, "a": a, "t": t, "b": b}


# ---------------------------------------------------------------------------
# Range clamp
# ---------------------------------------------------------------------------


class TestRangeClamp:
    def test_transposes_out_of_range_soprano_down_an_octave(self):
        # MIDI 90 is two semitones above hi=81 for soprano — one octave
        # down puts it at 78, comfortably in range.
        tokens = _voice_dict(
            s=[pitch_to_token(90), DUR_16TH],
            a=[pitch_to_token(60), DUR_16TH],
            t=[pitch_to_token(55), DUR_16TH],
            b=[pitch_to_token(40), DUR_16TH],
        )
        out = apply_voice_leading(tokens)
        assert token_to_pitch(out["s"][0]) == 78

    def test_transposes_out_of_range_bass_up_an_octave(self):
        # MIDI 20 is below bass lo=31; +12 = 32, which is in range.
        tokens = _voice_dict(
            s=[pitch_to_token(60), DUR_16TH],
            a=[pitch_to_token(55), DUR_16TH],
            t=[pitch_to_token(50), DUR_16TH],
            b=[pitch_to_token(20), DUR_16TH],
        )
        out = apply_voice_leading(tokens)
        assert token_to_pitch(out["b"][0]) == 32

    def test_noop_for_in_range_notes(self):
        tokens = _voice_dict(
            s=[pitch_to_token(60), DUR_16TH],
            a=[pitch_to_token(55), DUR_16TH],
            t=[pitch_to_token(50), DUR_16TH],
            b=[pitch_to_token(40), DUR_16TH],
        )
        # Disable coalesce + grid alignment — this test is isolating
        # the range-clamp behaviour on single-token streams.
        out = apply_voice_leading(
            tokens,
            bass_coalesce_min_sixteenths=0,
            upper_coalesce_min_sixteenths=0,
            align_grid_sixteenths=0,
        )
        for v in VOICES:
            assert out[v] == tokens[v]

    def test_passes_non_pitch_tokens_through(self):
        tokens = _voice_dict(
            s=[pitch_to_token(60), DUR_16TH, BAR, REST, DUR_16TH, PAD],
            a=[pitch_to_token(55), DUR_16TH, BAR, REST, DUR_16TH, PAD],
            t=[pitch_to_token(50), DUR_16TH, BAR, REST, DUR_16TH, PAD],
            b=[pitch_to_token(40), DUR_16TH, BAR, REST, DUR_16TH, PAD],
        )
        # Disable all rewriting passes — this test is isolating the
        # range-clamp behaviour, and per-voice coalesce / grid-align
        # would merge the 16th-note events.
        out = apply_voice_leading(
            tokens,
            bass_coalesce_min_sixteenths=0,
            upper_coalesce_min_sixteenths=0,
            align_grid_sixteenths=0,
        )
        for v in VOICES:
            # Non-pitch tokens unchanged at the same positions.
            assert out[v][1] == DUR_16TH
            assert out[v][2] == BAR
            assert out[v][3] == REST
            assert out[v][5] == PAD

    def test_respects_disable_flag(self):
        # 90 is out of soprano range; with the clamp disabled the token
        # should come back unchanged.
        tokens = _voice_dict(
            s=[pitch_to_token(90), DUR_16TH],
            a=[pitch_to_token(55), DUR_16TH],
            t=[pitch_to_token(50), DUR_16TH],
            b=[pitch_to_token(40), DUR_16TH],
        )
        out = apply_voice_leading(tokens, enable_range_clamp=False)
        assert token_to_pitch(out["s"][0]) == 90

    def test_all_ranges_match_constant(self):
        # Sanity: the tests assume SATB_RANGES values; if someone edits
        # the constant without updating these tests this catches it.
        assert SATB_RANGES == {
            "s": (55, 81),
            "a": (50, 76),
            "t": (47, 74),
            "b": (31, 64),
        }


# ---------------------------------------------------------------------------
# Parallel-motion detection
# ---------------------------------------------------------------------------


class TestParallelDetection:
    def test_flags_parallel_fifth_between_soprano_alto(self):
        # Step 0: S=67 A=60 (interval 7 — fifth). Step 1: S=69 A=62
        # (interval 7, both voices moved up 2 semitones). Parallel fifth.
        tokens = _voice_dict(
            s=[pitch_to_token(67), DUR_16TH, pitch_to_token(69), DUR_16TH],
            a=[pitch_to_token(60), DUR_16TH, pitch_to_token(62), DUR_16TH],
            t=[pitch_to_token(55), DUR_16TH, pitch_to_token(55), DUR_16TH],
            b=[pitch_to_token(40), DUR_16TH, pitch_to_token(40), DUR_16TH],
        )
        violations = detect_parallel_motion(tokens)
        # One violation: step 1, pair (s, a), interval 7.
        assert any(
            step == 1 and pair == ("s", "a") and interval == 7
            for step, pair, interval in violations
        )

    def test_flags_parallel_octave_between_tenor_bass(self):
        # Step 0: T=55 B=43 (interval 12). Step 1: T=57 B=45 (interval 12,
        # both up 2 semitones). Parallel octave.
        tokens = _voice_dict(
            s=[pitch_to_token(67), DUR_16TH, pitch_to_token(67), DUR_16TH],
            a=[pitch_to_token(60), DUR_16TH, pitch_to_token(60), DUR_16TH],
            t=[pitch_to_token(55), DUR_16TH, pitch_to_token(57), DUR_16TH],
            b=[pitch_to_token(43), DUR_16TH, pitch_to_token(45), DUR_16TH],
        )
        violations = detect_parallel_motion(tokens)
        assert any(
            pair == ("t", "b") and interval == 12 for _, pair, interval in violations
        )

    def test_ignores_contrary_motion(self):
        # S up, A down: classic contrary motion. No parallel violation
        # regardless of the interval match.
        tokens = _voice_dict(
            s=[pitch_to_token(67), DUR_16TH, pitch_to_token(69), DUR_16TH],
            a=[pitch_to_token(62), DUR_16TH, pitch_to_token(60), DUR_16TH],
            t=[pitch_to_token(55), DUR_16TH, pitch_to_token(55), DUR_16TH],
            b=[pitch_to_token(40), DUR_16TH, pitch_to_token(40), DUR_16TH],
        )
        assert detect_parallel_motion(tokens) == []

    def test_ignores_repeated_static_chord(self):
        # No voice moves. Even though intervals are 5ths/octaves, there's
        # no motion to be "parallel".
        tokens = _voice_dict(
            s=[pitch_to_token(67), DUR_16TH, pitch_to_token(67), DUR_16TH],
            a=[pitch_to_token(60), DUR_16TH, pitch_to_token(60), DUR_16TH],
            t=[pitch_to_token(55), DUR_16TH, pitch_to_token(55), DUR_16TH],
            b=[pitch_to_token(43), DUR_16TH, pitch_to_token(43), DUR_16TH],
        )
        assert detect_parallel_motion(tokens) == []

    def test_ignores_non_flagged_interval(self):
        # Interval = 4 semitones (major third) at both steps, same
        # direction. Thirds are fine in voice leading, not flagged.
        tokens = _voice_dict(
            s=[pitch_to_token(64), DUR_16TH, pitch_to_token(66), DUR_16TH],
            a=[pitch_to_token(60), DUR_16TH, pitch_to_token(62), DUR_16TH],
            t=[pitch_to_token(55), DUR_16TH, pitch_to_token(55), DUR_16TH],
            b=[pitch_to_token(40), DUR_16TH, pitch_to_token(40), DUR_16TH],
        )
        assert detect_parallel_motion(tokens) == []

    def test_apply_voice_leading_logs_warning_for_parallels(self, caplog):
        tokens = _voice_dict(
            s=[pitch_to_token(67), DUR_16TH, pitch_to_token(69), DUR_16TH],
            a=[pitch_to_token(60), DUR_16TH, pitch_to_token(62), DUR_16TH],
            t=[pitch_to_token(55), DUR_16TH, pitch_to_token(55), DUR_16TH],
            b=[pitch_to_token(40), DUR_16TH, pitch_to_token(40), DUR_16TH],
        )
        with caplog.at_level(logging.WARNING, logger="src.postprocess.voice_leading"):
            # Disable coalesce + grid-align — the 16th-note pairs this
            # test uses would otherwise get merged or snapped and there'd
            # be no moving voices to detect parallels between.
            apply_voice_leading(
                tokens,
                bass_coalesce_min_sixteenths=0,
                upper_coalesce_min_sixteenths=0,
                align_grid_sixteenths=0,
            )
        assert any("parallel fifth" in rec.message for rec in caplog.records)

    def test_disable_parallel_detect_produces_no_log(self, caplog):
        tokens = _voice_dict(
            s=[pitch_to_token(67), DUR_16TH, pitch_to_token(69), DUR_16TH],
            a=[pitch_to_token(60), DUR_16TH, pitch_to_token(62), DUR_16TH],
            t=[pitch_to_token(55), DUR_16TH, pitch_to_token(55), DUR_16TH],
            b=[pitch_to_token(40), DUR_16TH, pitch_to_token(40), DUR_16TH],
        )
        with caplog.at_level(logging.WARNING, logger="src.postprocess.voice_leading"):
            apply_voice_leading(tokens, enable_parallel_detect=False)
        assert caplog.records == []


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_missing_voice(self):
        tokens: dict[str, list[int]] = {"s": [], "a": [], "t": []}  # missing "b"
        import pytest

        with pytest.raises(ValueError, match="expected voices"):
            apply_voice_leading(tokens)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bass coalesce (issue 002)
# ---------------------------------------------------------------------------


def _dur(sixteenths: int) -> int:
    """Shortcut for a duration token at ``sixteenths`` (must be a bucket)."""
    assert sixteenths in DUR_BUCKETS, f"{sixteenths} not in DUR_BUCKETS {DUR_BUCKETS}"
    return duration_to_token(sixteenths)


class TestCoalesceVoiceTokens:
    def test_merges_adjacent_same_pitch_notes(self):
        # Two adjacent MIDI 43 quarter notes -> one half note.
        p = pitch_to_token(43)
        tokens = [SOS, p, _dur(4), p, _dur(4), EOS]
        out = coalesce_voice_tokens(tokens, min_sixteenths=1)
        # Re-decode: only one pitch event, totalling 8 sixteenths.
        assert out.count(p) == 1, out
        assert _dur(8) in out, out

    def test_absorbs_short_note_into_longer_neighbor(self):
        # 16th-note blip between two quarter-note neighbours — should vanish.
        p43 = pitch_to_token(43)
        p45 = pitch_to_token(45)
        tokens = [SOS, p43, _dur(4), p45, _dur(1), p43, _dur(4), EOS]
        out = coalesce_voice_tokens(tokens, min_sixteenths=4)
        # The p45 blip should have been absorbed and the two p43 neighbours
        # coalesced — expect a single p43 note of 9 sixteenths.
        assert p45 not in out, f"short blip should be gone; got {out}"

    def test_regenerates_bar_tokens_at_measure_boundaries(self):
        # Two whole notes (16 sixteenths each) — should produce exactly
        # one BAR token between them (both end at a measure boundary).
        p = pitch_to_token(43)
        tokens = [SOS, p, _dur(16), p, _dur(16), EOS]
        out = coalesce_voice_tokens(tokens, min_sixteenths=1)
        # Same-pitch merge gives one 32-sixteenth event, which splits into
        # two 16-sixteenth emissions across a bar line.
        assert out.count(BAR) == 1, out

    def test_does_not_mangle_rests(self):
        from src.data.vocab import REST
        p = pitch_to_token(43)
        tokens = [SOS, p, _dur(4), REST, _dur(4), p, _dur(4), EOS]
        out = coalesce_voice_tokens(tokens, min_sixteenths=1)
        # Rest must survive — same-pitch coalesce doesn't merge rest with note.
        assert REST in out, out


class TestCoalesceAllVoices:
    """Cover the generalised per-voice coalesce in ``apply_voice_leading``."""

    def test_upper_voices_also_coalesce(self):
        # All four voices share a "many 16ths" pattern; with default
        # upper_coalesce_min_sixteenths=2 the 16ths on S/A/T should get
        # absorbed just like they do on bass.
        def rapid_stream(midi: int) -> list[int]:
            p = pitch_to_token(midi)
            stream = [SOS]
            for _ in range(8):
                stream.extend([p, _dur(1)])  # 8 x 16th on the same pitch
            stream.append(EOS)
            return stream

        tokens = {
            "s": rapid_stream(60),
            "a": rapid_stream(55),
            "t": rapid_stream(50),
            "b": rapid_stream(40),
        }
        out = apply_voice_leading(tokens)
        # Each voice's 8 x 16th stream should coalesce into a single
        # whole-note event — the output should be shorter than the input
        # for every voice.
        for voice in ("s", "a", "t", "b"):
            assert len(out[voice]) < len(tokens[voice]), (
                f"{voice} should shorten after coalesce; before={len(tokens[voice])}, "
                f"after={len(out[voice])}"
            )

    def test_upper_coalesce_respects_threshold(self):
        # Upper voices default to min_sixteenths=2, bass to 3. A series
        # of dotted-8ths (3 sixteenths) should survive on bass but not
        # on the upper voices... wait, bass threshold is 3 so >= 3
        # survives; upper threshold is 2 so >= 2 also survives. Build a
        # single 16th (dur=1) on each voice — all should absorb it.
        def single_blip(midi: int) -> list[int]:
            p = pitch_to_token(midi)
            return [SOS, p, _dur(1), pitch_to_token(midi + 2), _dur(8), EOS]

        tokens = {
            "s": single_blip(60),
            "a": single_blip(55),
            "t": single_blip(50),
            "b": single_blip(40),
        }
        out = apply_voice_leading(tokens)
        # Each voice should end up with one *unique* pitch (the longer
        # neighbour) after the 16th is absorbed. The coalesce may split
        # a 9-sixteenth event into 8 + 1 across buckets, so the same
        # pitch token can appear more than once — what matters is that
        # the shorter neighbour's pitch is gone.
        for voice in ("s", "a", "t", "b"):
            pitches = {t for t in out[voice] if pitch_to_token(36) <= t <= pitch_to_token(80)}
            assert len(pitches) == 1, (
                f"{voice}: expected 1 unique pitch, got {pitches} in {out[voice]}"
            )

    def test_grid_align_aligns_event_ends_across_voices(self):
        # Two "voices" with different durations — after grid-snap to an
        # 8th-note grid (sixteenths=2) their cumulative event ends must
        # land on the same grid positions.
        s_events = [(60, 3), (62, 3), (64, 3)]   # ends at 3, 6, 9
        t_events = [(50, 5), (52, 5)]            # ends at 5, 10
        grid = 2
        s_aligned = _align_events_to_grid(s_events, grid)
        t_aligned = _align_events_to_grid(t_events, grid)

        def cumulative_ends(events):
            pos = 0
            out = []
            for _, d in events:
                pos += d
                out.append(pos)
            return out

        s_ends = cumulative_ends(s_aligned)
        t_ends = cumulative_ends(t_aligned)
        for e in s_ends + t_ends:
            assert e % grid == 0, f"end {e} not on grid {grid}"

    def test_grid_align_never_shrinks_event_to_zero(self):
        # A single 16th-note event on a quarter grid would round its end
        # to 0 if naïvely done; we must guarantee at least one quantum.
        events = [(60, 1)]
        aligned = _align_events_to_grid(events, grid_sixteenths=4)
        assert aligned[0][1] >= 4, aligned

    def test_coalesce_can_be_disabled_per_kind(self):
        # Raw 16th-note stream; disabling both coalesce paths should
        # leave the S/A/T and bass outputs alone (aside from identity
        # range-clamp / parallel-detect).
        def raw(midi: int) -> list[int]:
            p = pitch_to_token(midi)
            return [SOS, p, _dur(1), p, _dur(1), EOS]

        tokens = {
            "s": raw(60), "a": raw(55), "t": raw(50), "b": raw(40),
        }
        out = apply_voice_leading(
            tokens,
            bass_coalesce_min_sixteenths=0,
            upper_coalesce_min_sixteenths=0,
            align_grid_sixteenths=0,
        )
        for voice in ("s", "a", "t", "b"):
            assert out[voice] == tokens[voice], (
                f"{voice} should be unchanged when coalesce is disabled"
            )


# ---------------------------------------------------------------------------
# fit_to_length — exact-duration clamp applied as the final pipeline step.
# ---------------------------------------------------------------------------


def _total_sixteenths(tokens: list[int]) -> int:
    return sum(dur for _, dur in _tokens_to_events(tokens))


class TestFitToLength:
    def test_exact_match_leaves_duration_unchanged(self):
        # Two 8th notes = 4 sixteenths total. Asking for 4 must not
        # append or trim anything; round-tripping through the event
        # serializer is permitted (BAR / duration-decomposition), so
        # we assert on summed duration rather than byte-identical.
        p = pitch_to_token(60)
        tokens = [SOS, p, duration_to_token(2), p, duration_to_token(2), EOS]
        out = fit_to_length(tokens, 4)
        assert _total_sixteenths(out) == 4
        assert out[0] == SOS and out[-1] == EOS

    def test_pads_short_stream_with_rest(self):
        # Single 16th; target 16 → pad 15 sixteenths of rest.
        p = pitch_to_token(60)
        tokens = [SOS, p, duration_to_token(1), EOS]
        out = fit_to_length(tokens, 16)
        assert _total_sixteenths(out) == 16
        # A REST token must appear in the padded region.
        assert REST in out

    def test_trims_long_stream_to_target(self):
        # Four quarter-notes = 16 sixteenths. Target 10 → trim to 10.
        p = pitch_to_token(60)
        q = duration_to_token(4)
        tokens = [SOS, p, q, p, q, p, q, p, q, EOS]
        out = fit_to_length(tokens, 10)
        assert _total_sixteenths(out) == 10

    def test_trim_shortens_last_event_instead_of_dropping(self):
        # One half-note (8 sixteenths). Target 5 → keep the note, shorten
        # to 5 sixteenths (decomposed as 4 + 1 by the serializer).
        p = pitch_to_token(60)
        tokens = [SOS, p, duration_to_token(8), EOS]
        out = fit_to_length(tokens, 5)
        assert _total_sixteenths(out) == 5
        # Every emitted pitch token should still be the original pitch —
        # trimming must not invent new pitches.
        for pitch, _ in _tokens_to_events(out):
            assert pitch == 60

    def test_preserves_framing_when_absent(self):
        # No SOS/EOS in, no SOS/EOS out.
        p = pitch_to_token(60)
        tokens = [p, duration_to_token(2)]
        out = fit_to_length(tokens, 4)
        assert SOS not in out and EOS not in out
        assert _total_sixteenths(out) == 4

    def test_empty_input_pads_to_target(self):
        # No events at all → single REST of the target length.
        out = fit_to_length([], 6)
        assert _total_sixteenths(out) == 6
        assert REST in out

    def test_zero_target_returns_empty(self):
        # Target 0: every event must be trimmed away.
        p = pitch_to_token(60)
        tokens = [SOS, p, duration_to_token(4), EOS]
        out = fit_to_length(tokens, 0)
        assert _total_sixteenths(out) == 0

    def test_negative_target_raises(self):
        import pytest
        with pytest.raises(ValueError):
            fit_to_length([], -1)
