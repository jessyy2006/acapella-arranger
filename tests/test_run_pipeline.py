"""Tests for ``src.pipeline.run_pipeline`` orchestration.

The audio-extraction side lives in ``test_audio_pipeline.py``. This
module covers the orchestration layer:

- five-part Score output (issue 001)
- section-aware dispatch: one generation per unique label, with
  repeated labels paste-producing identical per-voice output
  (issue 003)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch

from src.data.vocab import DUR_OFFSET, DUR_BUCKETS, EOS, PITCH_OFFSET, SOS
from src.pipeline import run_pipeline as rp
from src.pipeline.run_pipeline import _assemble_score, _seed_for_label


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


# ---------------------------------------------------------------------------
# Section-aware dispatch (issue 003)
# ---------------------------------------------------------------------------


class _StubModel(torch.nn.Module):
    """Placeholder so ``run_pipeline`` has a model object to pass along —
    the tests fully monkey-patch ``generate_voice_tokens`` so this is
    never actually invoked.
    """

    def forward(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("stub model should not be invoked by the tests")


def _patched_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    sections: list[tuple[str, float, float]],
) -> tuple[Path, list[tuple[str, int | None]]]:
    """Drive ``run_pipeline`` with audio / model / audio-pipeline stubs.

    Records every ``generate_voice_tokens`` call as
    ``(voice, seed)`` so the tests can assert on call counts and seed
    determinism. Returns the output MIDI path + the recorded calls.
    """
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"\x00")
    ckpt_file = tmp_path / "ckpt.pt"
    ckpt_file.write_bytes(b"\x00")
    out_path = tmp_path / "out.mid"

    lead_tokens = _trivial_sequence() * 8  # some content for slicing to operate on
    # Normalise — _trivial_sequence ends with EOS, so repeat produces
    # multiple SOS/EOS interleaved. Strip intermediate ones.
    lead_tokens = [SOS] + [t for t in lead_tokens if t not in (SOS, EOS)] + [EOS]

    monkeypatch.setattr(
        rp, "extract_lead_tokens", lambda *_a, **_kw: (lead_tokens, 120.0),
    )
    monkeypatch.setattr(
        rp, "librosa",
        type("_Fake", (), {"load": staticmethod(lambda *_a, **_kw: (__import__("numpy").zeros(22050, dtype="float32"), 22050))}),
    )
    monkeypatch.setattr(rp, "detect_sections", lambda *_a, **_kw: sections)
    monkeypatch.setattr(rp, "_load_model", lambda *_a, **_kw: _StubModel())
    # Identity voice-leading so the stitched output isn't rewritten.
    monkeypatch.setattr(rp, "apply_voice_leading", lambda tokens, **_kw: tokens)

    calls: list[tuple[str, int]] = []

    def fake_generate(model, lead, voice, max_len, device, *, temperature, duration_temperature, top_k, seed=0):
        calls.append((voice, seed))
        # Return a per-(voice, seed) deterministic short sequence.
        return _trivial_sequence()

    monkeypatch.setattr(rp, "generate_voice_tokens", fake_generate)

    rp.run_pipeline(
        audio_path=audio_file,
        model_checkpoint=ckpt_file,
        out_path=out_path,
        voice_leading=True,
        device="cpu",
    )
    return out_path, calls


class TestSectionDispatch:
    def test_single_section_calls_generate_once_per_voice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With one detected section, the pipeline must invoke
        ``generate_voice_tokens`` exactly four times (once per voice).
        """
        out_path, calls = _patched_pipeline(
            monkeypatch, tmp_path, sections=[("A", 0.0, 4.0)],
        )
        assert out_path.is_file()
        voice_counts = Counter(v for v, _ in calls)
        assert voice_counts == {"s": 1, "a": 1, "t": 1, "b": 1}, voice_counts

    def test_repeated_sections_reuse_generated_tokens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With three sections (A, B, A), the pipeline must generate
        per *unique* label (2 labels × 4 voices = 8 calls), and the
        seed for label "A" must appear exactly once — proving the
        cache-by-label reuse path.
        """
        _, calls = _patched_pipeline(
            monkeypatch, tmp_path,
            sections=[("A", 0.0, 2.0), ("B", 2.0, 4.0), ("A", 4.0, 6.0)],
        )
        # Exactly 8 calls — four voices across the two unique labels.
        assert len(calls) == 8, calls
        # Seed for "A" must be consistent (one unique seed per label).
        seeds_by_voice: dict[str, set[int]] = {"s": set(), "a": set(), "t": set(), "b": set()}
        for voice, seed in calls:
            seeds_by_voice[voice].add(seed)
        for voice, seeds in seeds_by_voice.items():
            assert len(seeds) == 2, f"voice {voice} should see 2 unique seeds, got {seeds}"
        # And the per-label seed is the one derived by _seed_for_label.
        seed_a = _seed_for_label("A")
        seed_b = _seed_for_label("B")
        for voice in ("s", "a", "t", "b"):
            assert seed_a in seeds_by_voice[voice]
            assert seed_b in seeds_by_voice[voice]
