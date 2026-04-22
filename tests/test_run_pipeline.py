"""Tests for ``src.pipeline.run_pipeline`` orchestration.

The audio-extraction side lives in ``test_audio_pipeline.py``. This
module covers the orchestration layer — in particular that the
assembled Score exposes a five-part structure (Lead on top, then
S/A/T/B) as required by ``docs/issues/001-soprano-divergence-and-lead-track.md``.
"""

from __future__ import annotations

from src.data.vocab import DUR_OFFSET, DUR_BUCKETS, EOS, PITCH_OFFSET, SOS
from src.pipeline.run_pipeline import _assemble_score


def _trivial_sequence() -> list[int]:
    """SOS, middle-C, quarter-note duration, EOS — the smallest legal stream."""
    c4 = PITCH_OFFSET + 60
    quarter = DUR_OFFSET + DUR_BUCKETS.index(4)
    return [SOS, c4, quarter, EOS]


class TestAssembleScore:
    def test_score_has_five_parts_with_lead_on_top(self) -> None:
        lead_tokens = _trivial_sequence()
        generated = {v: _trivial_sequence() for v in ("s", "a", "t", "b")}

        score = _assemble_score(lead_tokens, generated)

        part_names = [p.partName for p in score.parts]
        assert len(score.parts) == 5, f"expected 5 parts, got {part_names}"
        assert part_names[0] == "LEAD", f"Lead should be top staff; got {part_names}"
        # Remaining order is the VOICES tuple rendered upper-case.
        assert part_names[1:] == ["S", "A", "T", "B"], part_names
