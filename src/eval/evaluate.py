"""Evaluate a model checkpoint on a cached test split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.loaders import collate_satb, load_dataset
from src.data.vocab import PAD
from src.eval.metrics import (
    bar_accuracy,
    duration_bucket_accuracy,
    per_voice_accuracy,
    pitch_range_compliance,
    voice_crossing_rate,
)
from src.models.baseline import SATBBaseline
from src.models.hybrid import SATBHybrid

logger = logging.getLogger(__name__)


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batch_to(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _build_model(model_class: str) -> torch.nn.Module:
    if model_class == "hybrid":
        return SATBHybrid()
    if model_class == "baseline":
        return SATBBaseline()
    raise ValueError(f"unknown model_class: {model_class!r}")


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    test_dataset_path: Path,
    *,
    batch_size: int = 32,
    device: str | None = None,
    model_class: str = "hybrid",
) -> dict[str, float]:
    """Return a flat dict mapping metric_name -> scalar."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(str(checkpoint_path))
    if not test_dataset_path.exists():
        raise FileNotFoundError(str(test_dataset_path))

    dev = _resolve_device(device)
    model = _build_model(model_class).to(dev).eval()

    state = torch.load(str(checkpoint_path), map_location="cpu")
    # Training saves state_dict; some callers may hand a full checkpoint dict.
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    ds = load_dataset(test_dataset_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_satb)

    acc_sum = {v: 0.0 for v in ("s", "a", "t", "b")}
    n_batches = 0
    dur_sum = 0.0
    bar_sum = 0.0
    bar_n = 0
    range_sum = {v: 0.0 for v in ("s", "a", "t", "b")}
    crossing_sum = 0.0
    crossing_n = 0

    for raw in loader:
        batch = _batch_to(raw, dev)
        logits = model(batch)
        targets = {v: batch[v] for v in ("s", "a", "t", "b")}

        per = per_voice_accuracy(logits, targets, pad_idx=PAD)
        for v in acc_sum:
            if np.isfinite(per[v]):
                acc_sum[v] += per[v]
        dur = duration_bucket_accuracy(logits, targets)
        if np.isfinite(dur):
            dur_sum += dur

        bacc = bar_accuracy(logits, targets)
        if np.isfinite(bacc):
            bar_sum += bacc
            bar_n += 1

        pred_tokens = {
            v: logits[v].argmax(dim=-1).detach().cpu().flatten().tolist() for v in ("s", "a", "t", "b")
        }
        pr = pitch_range_compliance(pred_tokens)
        for v in range_sum:
            if np.isfinite(pr[v]):
                range_sum[v] += pr[v]

        cr = voice_crossing_rate(pred_tokens)
        if np.isfinite(cr):
            crossing_sum += cr
            crossing_n += 1

        n_batches += 1

    out: dict[str, float] = {}
    for v in ("s", "a", "t", "b"):
        out[f"acc_{v}"] = acc_sum[v] / max(n_batches, 1)
        out[f"range_{v}"] = range_sum[v] / max(n_batches, 1)
    out["acc_mean"] = float(np.nanmean([out[f"acc_{v}"] for v in ("s", "a", "t", "b")]))
    out["duration_acc"] = dur_sum / max(n_batches, 1)
    out["bar_acc"] = bar_sum / max(bar_n, 1) if bar_n else float("nan")
    out["voice_crossing_rate"] = crossing_sum / max(crossing_n, 1) if crossing_n else float("nan")
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a SATB checkpoint on a cached split.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--model-class", choices=("hybrid", "baseline"), default="hybrid")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    args = _parse_args(argv)

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_dataset_path=args.split,
        batch_size=args.batch_size,
        device=args.device,
        model_class=args.model_class,
    )

    ckpt_name = args.checkpoint.stem
    out_json = args.out_json or (Path("reports") / f"{ckpt_name}_metrics.json")
    out_md = args.out_md or (Path("reports") / f"{ckpt_name}_metrics.md")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [f"## metrics: {ckpt_name}", "", "| metric | value |", "|---|---:|"]
    for k in sorted(metrics.keys()):
        v = metrics[k]
        s = "nan" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
        lines.append(f"| `{k}` | {s} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("wrote %s and %s", out_json, out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

