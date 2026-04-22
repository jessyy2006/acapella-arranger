"""Voice-leading-axis rubric evidence.

Generates N samples per (checkpoint, split) pair using the same sampled
autoregressive decoder as ``scripts/sample_midi.py``, runs
:func:`src.postprocess.voice_leading.detect_parallel_motion` on each
generated arrangement, and writes ``reports/parallels_count.md``
tabulating mean/median/max violation counts per checkpoint-split.

This is the quantitative backing for the voice-leading ablation axis:
since VL correction (range-clamp) is a no-op on hybrid model output
(range_compliance = 1.0), the meaningful VL metric is *"how often
does the model, with no help, produce textbook-forbidden parallel
5ths or 8ves?"* — which the detector counts directly.

Usage:
    python scripts/count_parallels.py --n 20
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_dataset
from src.eval.evaluate import _build_model, _load_hparams_from_sources, _resolve_device
from src.postprocess.voice_leading import detect_parallel_motion
from scripts.sample_midi import _generate_voice, _VOICES

logger = logging.getLogger("count_parallels")


@dataclass(frozen=True)
class Run:
    name: str
    checkpoint: Path
    split: Path
    model_class: str


_RUNS: tuple[Run, ...] = (
    Run("phase_a_jsb", Path("checkpoints/phase_a/phase_a_final.pt"),
        Path("data/processed/test_jsb.pt"), "hybrid"),
    Run("phase_a_jacappella", Path("checkpoints/phase_a/phase_a_final.pt"),
        Path("data/processed/test_jacappella.pt"), "hybrid"),
    Run("phase_b_jsb", Path("checkpoints/phase_b/phase_b_final.pt"),
        Path("data/processed/test_jsb.pt"), "hybrid"),
    Run("phase_b_jacappella", Path("checkpoints/phase_b/phase_b_final.pt"),
        Path("data/processed/test_jacappella.pt"), "hybrid"),
    Run("baseline_phase_a_jsb", Path("checkpoints/baseline_phase_a/baseline_phase_a_final.pt"),
        Path("data/processed/test_jsb.pt"), "baseline"),
)


def _load_model(ckpt_path: Path, model_class: str, device: torch.device) -> torch.nn.Module:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams = _load_hparams_from_sources(ckpt_path, state, model_class)
    if not hparams:
        raise ValueError(f"no hparams for {ckpt_path}")
    model = _build_model(model_class, hparams).to(device).eval()
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    return model


@torch.no_grad()
def count_run(
    run: Run,
    *,
    n_samples: int,
    max_len: int,
    temperature: float,
    duration_temperature: float,
    top_k: int,
    device: torch.device,
    lead_len_threshold: int = 20,
) -> dict[str, object]:
    """Return {per_sample_violations: list[int], summary: dict}."""
    logger.info("run=%s loading model...", run.name)
    model = _load_model(run.checkpoint, run.model_class, device)
    dataset = load_dataset(run.split)

    # Filter out degenerate examples (leads too short to generate from —
    # see Phase 1 finding that test_jacappella[0] is 8 bars of REST).
    usable_indices = [
        i for i in range(len(dataset))
        if dataset[i]["lead"].shape[0] >= lead_len_threshold
    ]
    if len(usable_indices) < n_samples:
        logger.warning(
            "only %d usable examples (>= %d tokens) in %s; taking all",
            len(usable_indices), lead_len_threshold, run.split,
        )
    indices = usable_indices[:n_samples]

    violations_per_sample: list[int] = []
    for idx in indices:
        lead = dataset[idx]["lead"].unsqueeze(0).to(device)
        generated: dict[str, list[int]] = {}
        for voice in _VOICES:
            generated[voice] = _generate_voice(
                model, lead, voice, max_len, device,
                temperature=temperature,
                duration_temperature=duration_temperature,
                top_k=top_k,
            )
        violations = detect_parallel_motion(generated)
        violations_per_sample.append(len(violations))
        logger.info("  sample %d (lead_len=%d): %d parallels", idx, lead.shape[1], len(violations))

    counts = violations_per_sample
    summary = {
        "n_samples": len(counts),
        "mean": statistics.mean(counts) if counts else float("nan"),
        "median": statistics.median(counts) if counts else float("nan"),
        "max": max(counts) if counts else 0,
        "min": min(counts) if counts else 0,
        "total": sum(counts),
    }
    return {"per_sample": counts, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="Examples per (checkpoint, split) pair.")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--duration-temperature", type=float, default=1.1)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-md", type=Path, default=Path("reports/parallels_count.md"))
    p.add_argument("--out-json", type=Path, default=Path("reports/parallels_count.json"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    device = _resolve_device(args.device)
    results: dict[str, dict] = {}
    for run in _RUNS:
        results[run.name] = count_run(
            run,
            n_samples=args.n,
            max_len=args.max_len,
            temperature=args.temperature,
            duration_temperature=args.duration_temperature,
            top_k=args.top_k,
            device=device,
        )

    # Write JSON first so partial results survive a later write error.
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {name: r for name, r in results.items()},
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    lines = [
        "## Parallel-motion violation counts per generated sample",
        "",
        f"Generated {args.n} samples per (checkpoint, test split) with "
        f"pitch-T={args.temperature}, dur-T={args.duration_temperature}, "
        f"top-k={args.top_k}. For each generated SATB arrangement, we "
        "count how many time-steps the voice-leading detector flags as "
        "parallel 5ths or parallel 8ves between adjacent voices. Lower "
        "is better.",
        "",
        "| run | mean | median | max | total |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in sorted(results):
        s = results[name]["summary"]
        lines.append(
            f"| `{name}` | {s['mean']:.2f} | {s['median']:.1f} | {s['max']} | {s['total']} |"
        )
    lines.extend([
        "",
        "### Interpretation",
        "",
        "A violation here is textbook-forbidden voice motion (two adjacent "
        "voices moving in the same direction while holding the same 5th or "
        "8ve interval). The model was not explicitly trained to avoid these, "
        "so the count is a direct measure of how well the model internalised "
        "the constraint from the training data. The detector is implemented "
        "by `src.postprocess.voice_leading.detect_parallel_motion` and "
        "defined formally in that module's docstring.",
    ])
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s and %s", args.out_md, args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
