"""CLI training loop for the SATB harmony model (hybrid or baseline).

Usage
-----
    python -m src.training.train --config configs/train.yaml --phase pretrain
    python -m src.training.train --config configs/train.yaml --phase finetune \
        --init-from checkpoints/phase_a/best.pt
    python -m src.training.train --config configs/train.yaml --dry-run

Designed to be invoked from a Colab notebook: auto-resumes from
``<ckpt_dir>/last.pt`` on restart, flushes a checkpoint on SIGTERM/SIGINT
so idle-kill doesn't lose the in-flight epoch, and stays entirely in CSV
+ matplotlib for artifacts (no wandb / tensorboard dependency).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import signal
import sys
from dataclasses import replace
from pathlib import Path
from types import FrameType
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")  # headless backend — Colab has no display
import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.loaders import collate_satb, load_dataset  # noqa: E402
from src.data.vocab import PAD  # noqa: E402
from src.models.baseline import SATBBaseline  # noqa: E402
from src.models.hybrid import SATBHybrid  # noqa: E402
from src.training.checkpoint import (  # noqa: E402
    BEST_FILENAME,
    LAST_FILENAME,
    build_state,
    resume_or_init,
    save,
)
from src.training.config import (  # noqa: E402
    ModelCfg,
    TrainConfig,
    config_to_dict,
    load_config,
    resolve_device,
)
from src.training.metrics import compute_accuracy, compute_loss  # noqa: E402

logger = logging.getLogger("src.training.train")


# ---------------------------------------------------------------------------
# Seeding + device helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed torch (cpu+cuda), numpy, and python stdlib random."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _batch_to(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move every tensor in the collate dict to ``device`` (preserves non-tensors)."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ---------------------------------------------------------------------------
# Model + data factory
# ---------------------------------------------------------------------------


def build_model(model_cfg: ModelCfg) -> nn.Module:
    """Dispatch to SATBHybrid or SATBBaseline based on config string."""
    common = dict(
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_encoder_layers=model_cfg.n_encoder_layers,
        d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout,
    )
    if model_cfg.cls == "hybrid":
        return SATBHybrid(
            **common,
            n_decoder_lstm_layers=model_cfg.n_decoder_lstm_layers,
        )
    if model_cfg.cls == "baseline":
        return SATBBaseline(
            **common,
            n_decoder_layers=model_cfg.n_decoder_layers,
        )
    raise ValueError(f"unknown model class: {model_cfg.cls!r} (expected hybrid|baseline)")


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Load cached train + val SATBDatasets and wrap in DataLoaders.

    ``num_workers`` drops to 0 for CPU runs (``--dry-run`` on dev machines)
    because macOS multiprocessing with music21 in the dataset triggers
    spurious fork warnings and is strictly slower than single-process
    loading on small batches.
    """
    train_ds = load_dataset(cfg.data.processed_file)
    val_ds = load_dataset(cfg.data.val_file)

    workers = cfg.num_workers if torch.cuda.is_available() else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_satb,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_satb,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Train / validate steps
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    max_grad_norm: float,
    log_every: int,
    epoch: int,
) -> dict[str, float]:
    """One epoch of training. Returns averaged metrics for the CSV row."""
    model.train()
    running: dict[str, float] = {"total": 0.0, "s": 0.0, "a": 0.0, "t": 0.0, "b": 0.0}
    n_batches = 0

    use_amp = scaler is not None and device.type == "cuda"

    for step, raw_batch in enumerate(loader):
        batch = _batch_to(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(batch)
                loss, per_voice = compute_loss(logits, batch, pad_idx=PAD)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch)
            loss, per_voice = compute_loss(logits, batch, pad_idx=PAD)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        running["total"] += loss.detach().item()
        for v, x in per_voice.items():
            running[v] += x
        n_batches += 1

        if step % log_every == 0:
            logger.info(
                "epoch=%d step=%d loss=%.4f (s=%.3f a=%.3f t=%.3f b=%.3f)",
                epoch,
                step,
                loss.detach().item(),
                per_voice["s"],
                per_voice["a"],
                per_voice["t"],
                per_voice["b"],
            )

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Single validation pass. Returns averaged loss + per-voice accuracy."""
    model.eval()
    running: dict[str, float] = {"total": 0.0, "s": 0.0, "a": 0.0, "t": 0.0, "b": 0.0}
    acc_sum: dict[str, float] = {"s": 0.0, "a": 0.0, "t": 0.0, "b": 0.0}
    n_batches = 0

    for raw_batch in loader:
        batch = _batch_to(raw_batch, device)
        logits = model(batch)
        loss, per_voice = compute_loss(logits, batch, pad_idx=PAD)
        accs = compute_accuracy(logits, batch, pad_idx=PAD)
        running["total"] += loss.item()
        for v, x in per_voice.items():
            running[v] += x
        for v, a in accs.items():
            acc_sum[v] += a
        n_batches += 1

    if n_batches == 0:
        # Edge case: --limit-jsb / --limit-jacappella shrinks val to zero batches.
        return {"total": float("nan"), **{v: float("nan") for v in ("s", "a", "t", "b")},
                **{f"acc_{v}": float("nan") for v in ("s", "a", "t", "b")}}

    out = {k: v / n_batches for k, v in running.items()}
    out.update({f"acc_{v}": a / n_batches for v, a in acc_sum.items()})
    return out


# ---------------------------------------------------------------------------
# CSV + plotting
# ---------------------------------------------------------------------------


_CSV_HEADER: tuple[str, ...] = (
    "epoch",
    "train_total",
    "train_s",
    "train_a",
    "train_t",
    "train_b",
    "val_total",
    "val_s",
    "val_a",
    "val_t",
    "val_b",
    "val_acc_s",
    "val_acc_a",
    "val_acc_t",
    "val_acc_b",
)


def _open_csv(path: Path, append: bool) -> tuple[Any, Any]:
    """Open the metrics CSV, writing the header only on a fresh run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    handle = path.open(mode, newline="")
    writer = csv.writer(handle)
    if not append:
        writer.writerow(_CSV_HEADER)
        handle.flush()
    return handle, writer


def _csv_row(epoch: int, train: dict[str, float], val: dict[str, float]) -> list[Any]:
    return [
        epoch,
        train["total"], train["s"], train["a"], train["t"], train["b"],
        val.get("total", float("nan")),
        val.get("s", float("nan")), val.get("a", float("nan")),
        val.get("t", float("nan")), val.get("b", float("nan")),
        val.get("acc_s", float("nan")), val.get("acc_a", float("nan")),
        val.get("acc_t", float("nan")), val.get("acc_b", float("nan")),
    ]


def plot_loss_curves(csv_path: Path, png_path: Path) -> None:
    """Two-panel plot: summed total loss + per-voice breakdown."""
    epochs: list[int] = []
    train_total: list[float] = []
    val_total: list[float] = []
    per_voice_train: dict[str, list[float]] = {v: [] for v in ("s", "a", "t", "b")}
    per_voice_val: dict[str, list[float]] = {v: [] for v in ("s", "a", "t", "b")}

    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_total.append(float(row["train_total"]))
            val_total.append(float(row["val_total"]))
            for v in ("s", "a", "t", "b"):
                per_voice_train[v].append(float(row[f"train_{v}"]))
                per_voice_val[v].append(float(row[f"val_{v}"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_total, label="train")
    ax1.plot(epochs, val_total, label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("summed CE")
    ax1.set_title("Total loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for v in ("s", "a", "t", "b"):
        ax2.plot(epochs, per_voice_val[v], label=f"val {v}")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("per-voice val CE")
    ax2.set_title("Per-voice validation loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the SATB harmony model (hybrid or baseline).",
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--phase", choices=("pretrain", "finetune"), default=None)
    p.add_argument("--init-from", type=Path, default=None)
    p.add_argument("--model-class", choices=("hybrid", "baseline"), default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    args = _parse_args(argv)

    cfg = load_config(args.config, phase=args.phase)
    if args.model_class is not None:
        cfg = replace(cfg, model=replace(cfg.model, cls=args.model_class))

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    # Build model + data
    model = build_model(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model=%s params=%s device=%s", cfg.model.cls, f"{n_params:,}", device)
    logger.info("config: %s", json.dumps(config_to_dict(cfg), indent=None))

    train_loader, val_loader = build_dataloaders(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler.factor,
        patience=cfg.scheduler.patience,
        min_lr=cfg.scheduler.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.type == "cuda") else None

    # Resume / init
    ckpt_dir = Path(cfg.checkpoint.dir) / cfg.checkpoint.run_name
    start_epoch, best_val_loss = resume_or_init(
        ckpt_dir=ckpt_dir,
        init_from=args.init_from,
        allow_resume=not args.no_resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    resumed = start_epoch > 0

    # --- dry-run: one forward+backward+step on one batch, then exit ---
    if args.dry_run:
        model.train()
        batch = _batch_to(next(iter(train_loader)), device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss, _per_voice = compute_loss(logits, batch, pad_idx=PAD)
        if not torch.isfinite(loss):
            logger.error("dry-run loss is non-finite: %s", loss)
            return 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
        optimizer.step()
        logger.info("dry-run ok: loss=%.4f", loss.detach().item())
        return 0

    # --- signal handler: flush last.pt before exit so Colab kill doesn't lose work ---
    def _flush_and_exit(signum: int, _frame: FrameType | None) -> None:
        logger.warning("caught signal %d — flushing last.pt before exit", signum)
        state = build_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=current_epoch["value"],
            best_val_loss=best_val["value"],
        )
        save(state, ckpt_dir / LAST_FILENAME)
        sys.exit(128 + signum)

    # Mutable holders so the signal closure sees live values.
    current_epoch: dict[str, int] = {"value": start_epoch - 1}
    best_val: dict[str, float] = {"value": best_val_loss}

    signal.signal(signal.SIGTERM, _flush_and_exit)
    signal.signal(signal.SIGINT, _flush_and_exit)

    # --- CSV ---
    csv_path = Path(cfg.reports.dir) / f"{cfg.checkpoint.run_name}_loss.csv"
    csv_handle, csv_writer = _open_csv(csv_path, append=resumed)

    # --- Main training loop ---
    early_stop_counter = 0
    try:
        for epoch in range(start_epoch, cfg.epochs):
            current_epoch["value"] = epoch
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                max_grad_norm=cfg.optim.max_grad_norm,
                log_every=cfg.log_every,
                epoch=epoch,
            )

            val_metrics: dict[str, float] = {}
            if (epoch % cfg.val_every) == 0:
                val_metrics = validate(model, val_loader, device)
                logger.info(
                    "epoch=%d val_loss=%.4f (s=%.3f a=%.3f t=%.3f b=%.3f)",
                    epoch,
                    val_metrics["total"],
                    val_metrics["s"], val_metrics["a"],
                    val_metrics["t"], val_metrics["b"],
                )
                if np.isfinite(val_metrics["total"]):
                    scheduler.step(val_metrics["total"])
                improved = val_metrics["total"] < best_val["value"] - cfg.early_stopping.min_delta
                if improved:
                    best_val["value"] = val_metrics["total"]
                    save(
                        build_state(
                            model=model, optimizer=optimizer, scheduler=scheduler,
                            epoch=epoch, best_val_loss=best_val["value"],
                        ),
                        ckpt_dir / BEST_FILENAME,
                    )
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

            csv_writer.writerow(_csv_row(epoch, train_metrics, val_metrics))
            csv_handle.flush()

            save_due = (epoch % cfg.checkpoint.save_every) == 0 or epoch == cfg.epochs - 1
            if save_due:
                save(
                    build_state(
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch, best_val_loss=best_val["value"],
                    ),
                    ckpt_dir / LAST_FILENAME,
                )

            if early_stop_counter >= cfg.early_stopping.patience:
                logger.info(
                    "early stopping at epoch=%d (no improvement for %d val passes)",
                    epoch, early_stop_counter,
                )
                break
    finally:
        csv_handle.close()

    # --- Final artifacts ---
    final_path = ckpt_dir / f"{cfg.checkpoint.run_name}_final.pt"
    best_path = ckpt_dir / BEST_FILENAME
    if best_path.exists():
        shutil.copy2(best_path, final_path)
        logger.info("wrote final checkpoint: %s", final_path)

    png_path = Path(cfg.reports.dir) / f"{cfg.checkpoint.run_name}_loss.png"
    if csv_path.exists():
        plot_loss_curves(csv_path, png_path)
        logger.info("wrote loss curves: %s", png_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
