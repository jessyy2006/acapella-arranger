"""Tests for ``src.pipeline.sections``.

Cover the three spec-required cases from
``docs/issues/003-satb-voices-lack-sectional-pattern-structure.md``:

- ``detect_sections`` covers ``[0, duration]`` with non-overlapping
  segments;
- ``detect_sections`` falls back to a single label on short audio;
- ``tokens_for_section_window`` slices the lead token stream correctly
  for a given time window.
"""

from __future__ import annotations

import numpy as np

from src.data.vocab import (
    BAR,
    DUR_BUCKETS,
    DUR_OFFSET,
    EOS,
    PITCH_OFFSET,
    SOS,
    duration_to_token,
    pitch_to_token,
)
from src.pipeline.sections import detect_sections, tokens_for_section_window


def _noise(duration_sec: float, sr: int = 22_050, seed: int = 0) -> np.ndarray:
    """Low-volume pink-ish noise long enough for librosa to segment."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration_sec)
    return rng.standard_normal(n).astype(np.float32) * 0.1


class TestDetectSections:
    def test_returns_covering_non_overlapping_segments(self):
        sr = 22_050
        # Two clearly-different halves: white noise then a sine tone.
        half_dur = 4.0
        noise = _noise(half_dur, sr)
        t = np.arange(int(sr * half_dur)) / sr
        tone = 0.2 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        audio = np.concatenate([noise, tone])

        sections = detect_sections(audio, sr)

        assert len(sections) >= 1, sections
        # Covers the full duration with no gap / overlap.
        assert sections[0][1] == 0.0
        expected_end = len(audio) / sr
        assert abs(sections[-1][2] - expected_end) < 0.1, sections
        for i in range(len(sections) - 1):
            assert sections[i][2] == sections[i + 1][1], (
                f"gap/overlap at {i}: {sections[i]} -> {sections[i + 1]}"
            )
        # Every label should be a single uppercase ASCII letter.
        for label, _, _ in sections:
            assert label.isalpha() and label == label.upper() and len(label) == 1

    def test_falls_back_to_single_label_on_short_audio(self):
        # Below ``2 * min_section_sec`` the helper must return one section
        # labelled "A" covering the full duration — makes the downstream
        # pipeline degrade to the pre-section-aware behaviour on short
        # clips like ``outputs/smoke_input.mp3``.
        sr = 22_050
        audio = _noise(1.0, sr)  # 1.0 s < 2 * 2.0 s default min_section_sec

        sections = detect_sections(audio, sr)

        assert len(sections) == 1, sections
        label, start, end = sections[0]
        assert label == "A"
        assert start == 0.0
        assert abs(end - 1.0) < 0.01


class TestTokensForSectionWindow:
    def _build_stream(self, events_per_bar: int = 4, num_bars: int = 4) -> list[int]:
        """Build ``num_bars`` bars of 16ths (``events_per_bar * 4`` events)
        at MIDI 60 — every event is a quarter note to keep math simple.
        """
        quarter_dur = duration_to_token(4)  # 4 sixteenths = quarter note
        pitch = pitch_to_token(60)
        tokens: list[int] = [SOS]
        for _ in range(num_bars):
            for _ in range(events_per_bar):
                tokens.extend([pitch, quarter_dur])
            tokens.append(BAR)
        tokens.append(EOS)
        return tokens

    def test_slices_correct_window_at_four_four(self):
        # 120 bpm → one 16th = 0.125 s → one bar (4/4) = 4 quarters = 2 s.
        # A 2-bar window [0 s, 4 s] should land on the first two bars.
        tokens = self._build_stream(events_per_bar=4, num_bars=4)
        start_idx, end_idx = tokens_for_section_window(tokens, 120.0, 0.0, 4.0)
        sliced = tokens[start_idx:end_idx]
        # 2 bars * 4 events per bar * 2 tokens per event = 16 pitch+dur tokens.
        pitch_plus_dur_count = sum(1 for t in sliced if t not in (SOS, EOS, BAR))
        assert pitch_plus_dur_count == 16, f"expected 16 content tokens, got {pitch_plus_dur_count}"

    def test_empty_window_returns_empty_slice(self):
        tokens = self._build_stream(events_per_bar=4, num_bars=4)
        assert tokens_for_section_window(tokens, 120.0, 100.0, 200.0) == (0, 0)

    def test_degenerate_tempo_returns_empty(self):
        tokens = self._build_stream(events_per_bar=4, num_bars=4)
        assert tokens_for_section_window(tokens, 0.0, 0.0, 4.0) == (0, 0)
