"""Checkpoint save/load + resume logic that survives Colab disconnects.

Checkpoints contain only tensor state dicts (no pickled Python objects),
so ``torch.load(..., weights_only=True)`` is both safe and strict — same
hardening pattern as :func:`src.data.loaders.load_dataset`.

RNG state is intentionally **not** saved. ``np.random.get_state()`` and
``random.getstate()`` return plain Python tuples that ``weights_only=True``
refuses to unpickle. Trading byte-level RNG continuity on resume for a
strict load-time guard is the right call: loss-curve shape, optimizer
state, and model weights all round-trip, which is what the spec's
acceptance criteria demand.

Resume priority is deliberate and logged loudly:

    last.pt in checkpoint dir  >  --init-from path  >  fresh init

This means a Colab disconnect mid-Phase-B transparently resumes Phase B
rather than silently dropping back to the Phase A weights specified by
``--init-from``. Operators debug disconnects by grepping the startup log
line that names which branch fired.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

LAST_FILENAME = "last.pt"
BEST_FILENAME = "best.pt"


def build_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_loss: float,
) -> dict[str, Any]:
    """Bundle the model + training state into a dict ready for ``torch.save``."""
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }


def save(state: dict[str, Any], path: Path) -> None:
    """Atomic-ish save: write to ``<path>.tmp`` then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load(path: Path) -> dict[str, Any]:
    """Load a checkpoint with ``weights_only=True`` (state-dict-only format)."""
    return torch.load(str(path), map_location="cpu", weights_only=True)


def find_latest(ckpt_dir: Path) -> Path | None:
    """Return the path to ``last.pt`` if it exists, else ``None``."""
    candidate = Path(ckpt_dir) / LAST_FILENAME
    return candidate if candidate.exists() else None


def resume_or_init(
    *,
    ckpt_dir: Path,
    init_from: Path | None,
    allow_resume: bool,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> tuple[int, float]:
    """Apply the resume-vs-init-from-vs-fresh priority rule.

    Returns ``(start_epoch, best_val_loss)``. ``start_epoch`` is 0 for a
    fresh or ``--init-from`` run; otherwise it's the epoch the resumed
    checkpoint finished, and the main loop picks up from ``start_epoch``.

    A log line at INFO level names the branch taken so operators can
    diagnose Colab disconnects from the training stdout alone.
    """
    ckpt_dir = Path(ckpt_dir)
    latest = find_latest(ckpt_dir) if allow_resume else None

    if latest is not None:
        state = load(latest)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state["best_val_loss"])
        logger.info(
            "resuming from %s: start_epoch=%d, best_val_loss=%.4f",
            latest,
            start_epoch,
            best_val,
        )
        return start_epoch, best_val

    if init_from is not None:
        state = load(Path(init_from))
        model.load_state_dict(state["model_state"])
        logger.info(
            "initialised weights from %s (optimizer/epoch fresh)", init_from
        )
        return 0, float("inf")

    logger.info("fresh init: no checkpoint at %s, no --init-from", ckpt_dir)
    return 0, float("inf")
