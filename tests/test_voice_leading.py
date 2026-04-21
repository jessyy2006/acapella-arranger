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
from src.postprocess.voice_leading import (
    apply_voice_leading,
    detect_parallel_motion,
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
        out = apply_voice_leading(tokens)
        for v in VOICES:
            assert out[v] == tokens[v]

    def test_passes_non_pitch_tokens_through(self):
        tokens = _voice_dict(
            s=[pitch_to_token(60), DUR_16TH, BAR, REST, DUR_16TH, PAD],
            a=[pitch_to_token(55), DUR_16TH, BAR, REST, DUR_16TH, PAD],
            t=[pitch_to_token(50), DUR_16TH, BAR, REST, DUR_16TH, PAD],
            b=[pitch_to_token(40), DUR_16TH, BAR, REST, DUR_16TH, PAD],
        )
        out = apply_voice_leading(tokens)
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
            apply_voice_leading(tokens)
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
