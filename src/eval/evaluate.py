"""Evaluate a model checkpoint on a cached test split.

Resolves model hyperparameters in priority order:
    1. ``hparams`` key inside the checkpoint dict (if the training loop
       saved one).
    2. A sibling ``<checkpoint-stem>.config.json`` next to the ``.pt``.
    3. ``configs/train.yaml`` in the repo root, loaded via
       :func:`src.training.config.load_config`.

Without this, ``_build_model`` used default constructor args while training
used the yaml-configured ones — ``load_state_dict`` would silently mis-load
and every metric came out garbage (the bug that produced ``acc_s=0.28``
against a trained checkpoint whose training CSV reported ``val_acc_s=0.63``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.loaders import collate_satb, load_dataset
from src.data.vocab import PAD
from src.eval.metrics import (
    bar_accuracy,
    duration_bucket_accuracy,
    per_voice_accuracy,
    pitch_count_by_voice,
    pitch_range_compliance,
    voice_crossing_rate,
)
from src.models.baseline import SATBBaseline
from src.models.hybrid import SATBHybrid
from src.training.config import load_config

logger = logging.getLogger(__name__)

_VOICES: tuple[str, ...] = ("s", "a", "t", "b")

_HYBRID_HPARAM_KEYS: tuple[str, ...] = (
    "d_model",
    "n_heads",
    "n_encoder_layers",
    "n_decoder_lstm_layers",
    "d_ff",
    "dropout",
)
_BASELINE_HPARAM_KEYS: tuple[str, ...] = (
    "d_model",
    "n_heads",
    "n_encoder_layers",
    "n_decoder_layers",
    "d_ff",
    "dropout",
)


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batch_to(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _filter_hparams(raw: dict[str, Any], model_class: str) -> dict[str, Any]:
    keys = _HYBRID_HPARAM_KEYS if model_class == "hybrid" else _BASELINE_HPARAM_KEYS
    return {k: raw[k] for k in keys if k in raw}


def _load_hparams_from_sources(
    checkpoint_path: Path,
    checkpoint_obj: Any,
    model_class: str,
) -> dict[str, Any]:
    """Walk the priority list of hparam sources. Returns kwargs for the
    constructor, or an empty dict if nothing is found (caller decides
    whether that's fatal)."""
    if isinstance(checkpoint_obj, dict) and isinstance(checkpoint_obj.get("hparams"), dict):
        logger.info("loading hparams from checkpoint 'hparams' key")
        return _filter_hparams(checkpoint_obj["hparams"], model_class)

    sibling = checkpoint_path.with_suffix(".config.json")
    if sibling.exists():
        logger.info("loading hparams from sibling config: %s", sibling)
        return _filter_hparams(json.loads(sibling.read_text(encoding="utf-8")), model_class)

    yaml_path = Path("configs/train.yaml")
    if yaml_path.exists():
        logger.info("loading hparams from %s (fallback — no sidecar config found)", yaml_path)
        cfg = load_config(yaml_path)
        model_cfg = cfg.model
        raw = {
            "d_model": model_cfg.d_model,
            "n_heads": model_cfg.n_heads,
            "n_encoder_layers": model_cfg.n_encoder_layers,
            "n_decoder_lstm_layers": model_cfg.n_decoder_lstm_layers,
            "n_decoder_layers": model_cfg.n_decoder_layers,
            "d_ff": model_cfg.d_ff,
            "dropout": model_cfg.dropout,
        }
        return _filter_hparams(raw, model_class)

    return {}


def _build_model(
    model_class: str,
    hparams: dict[str, Any] | None = None,
) -> torch.nn.Module:
    kwargs = hparams or {}
    if model_class == "hybrid":
        return SATBHybrid(**kwargs)
    if model_class == "baseline":
        return SATBBaseline(**kwargs)
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

    # weights_only=False is required because the training loop saves a
    # dict with optimizer/scheduler state alongside the model state. PyTorch
    # 2.6 changed the default to True, which refuses anything but raw
    # tensors. See https://pytorch.org/docs/stable/generated/torch.load.html.
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    hparams = _load_hparams_from_sources(checkpoint_path, state, model_class)
    if not hparams:
        raise ValueError(
            f"no hparams found for {checkpoint_path.name}: checkpoint lacks 'hparams' key, "
            f"no sibling {checkpoint_path.stem}.config.json exists, and configs/train.yaml is "
            "not available from the current working directory. Pass hparams another way or run "
            "from the repo root."
        )

    dev = _resolve_device(device)
    model = _build_model(model_class, hparams).to(dev).eval()

    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    ds = load_dataset(test_dataset_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_satb)

    acc_sum = {v: 0.0 for v in _VOICES}
    n_batches = 0
    dur_sum = 0.0
    bar_sum = 0.0
    bar_n = 0
    range_sum = {v: 0.0 for v in _VOICES}
    range_batches = {v: 0 for v in _VOICES}
    crossing_sum = 0.0
    crossing_n = 0
    pitch_count_total = {v: 0 for v in _VOICES}

    for raw in loader:
        batch = _batch_to(raw, dev)
        logits = model(batch)
        targets = {v: batch[v] for v in _VOICES}

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

        # Per-sample (batch_size, L-1) list-of-lists — NOT flattened. The
        # metrics below need to keep sample boundaries to reconstruct
        # per-song timelines / per-song pitch counts.
        pred_lists = {
            v: logits[v][:, :-1, :].argmax(dim=-1).detach().cpu().tolist() for v in _VOICES
        }

        pr = pitch_range_compliance(pred_lists)
        for v in range_sum:
            if np.isfinite(pr[v]):
                range_sum[v] += pr[v]
                range_batches[v] += 1
        pc = pitch_count_by_voice(pred_lists)
        for v in _VOICES:
            pitch_count_total[v] += pc[v]

        cr = voice_crossing_rate(pred_lists)
        if np.isfinite(cr):
            crossing_sum += cr
            crossing_n += 1

        n_batches += 1

    out: dict[str, float] = {}
    for v in _VOICES:
        out[f"acc_{v}"] = acc_sum[v] / max(n_batches, 1)
        out[f"range_{v}"] = (
            range_sum[v] / range_batches[v] if range_batches[v] else float("nan")
        )
        out[f"pitch_count_{v}"] = float(pitch_count_total[v])
    out["acc_mean"] = float(np.nanmean([out[f"acc_{v}"] for v in _VOICES]))
    out["duration_acc"] = dur_sum / max(n_batches, 1)
    out["bar_acc"] = bar_sum / bar_n if bar_n else float("nan")
    out["voice_crossing_rate"] = crossing_sum / crossing_n if crossing_n else float("nan")
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
