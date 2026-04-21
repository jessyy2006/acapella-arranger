"""Typed training config + YAML loader with phase-override merge.

A single ``configs/train.yaml`` backs every run — pretrain, fine-tune, and
ablations. The file has a base section plus a ``phases`` block with named
override dicts. :func:`load_config` deep-merges the selected phase into the
base and constructs a frozen :class:`TrainConfig`, so code downstream can
rely on every field being present and of the right type (no ``cfg.get(...)``
scattered through the training loop).

Unknown keys raise ``TypeError`` on dataclass construction. This is
intentional — a silent ignore of a mistyped key in a training config is
the exact failure mode that wastes a multi-hour Colab run.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCfg:
    cls: str = "hybrid"  # "hybrid" | "baseline"
    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 3
    n_decoder_lstm_layers: int = 2  # hybrid only
    n_decoder_layers: int = 2  # baseline only
    d_ff: int = 1024
    dropout: float = 0.1


@dataclass(frozen=True)
class DataCfg:
    processed_file: str = "data/processed/train.pt"
    val_file: str = "data/processed/val.pt"


@dataclass(frozen=True)
class OptimCfg:
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class SchedulerCfg:
    name: str = "reduce_on_plateau"
    factor: float = 0.5
    patience: int = 3
    min_lr: float = 1e-6


@dataclass(frozen=True)
class EarlyStopCfg:
    patience: int = 6
    min_delta: float = 1e-4


@dataclass(frozen=True)
class CheckpointCfg:
    dir: str = "checkpoints"
    run_name: str = "run"
    save_every: int = 5


@dataclass(frozen=True)
class ReportsCfg:
    dir: str = "reports"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    epochs: int = 25
    batch_size: int = 16
    num_workers: int = 2
    amp: bool = True
    log_every: int = 50
    val_every: int = 1

    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    scheduler: SchedulerCfg = field(default_factory=SchedulerCfg)
    early_stopping: EarlyStopCfg = field(default_factory=EarlyStopCfg)
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    reports: ReportsCfg = field(default_factory=ReportsCfg)


# Map from top-level config keys to the dataclass that represents them.
# Used to dispatch nested dict payloads to the right constructor.
_NESTED_SECTIONS: dict[str, type] = {
    "model": ModelCfg,
    "data": DataCfg,
    "optim": OptimCfg,
    "scheduler": SchedulerCfg,
    "early_stopping": EarlyStopCfg,
    "checkpoint": CheckpointCfg,
    "reports": ReportsCfg,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — override wins on leaves, nested dicts merge.

    Both inputs are left untouched; returns a fresh dict.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build_section(section_cls: type, payload: dict[str, Any]) -> Any:
    """Construct a nested dataclass from its YAML dict payload."""
    if not isinstance(payload, dict):
        raise TypeError(
            f"expected a mapping for {section_cls.__name__}, got {type(payload).__name__}"
        )
    # YAML uses "class" (a reserved word in Python) as the model-class key;
    # map it to the dataclass's ``cls`` field so users don't have to quote.
    # Copy first so we don't mutate the caller's payload dict.
    payload = dict(payload)
    if section_cls is ModelCfg and "class" in payload:
        payload["cls"] = payload.pop("class")
    return section_cls(**payload)


def load_config(path: str | Path, phase: str | None = None) -> TrainConfig:
    """Load YAML, strip ``phases`` block, deep-merge the selected phase, build.

    Parameters
    ----------
    path
        Path to the YAML file.
    phase
        Phase override to apply. ``None`` uses the base section as-is.
        Non-``None`` phase must exist under ``phases:`` in the YAML.

    Raises
    ------
    FileNotFoundError
        If ``path`` doesn't exist.
    KeyError
        If ``phase`` is given but not defined in the YAML's ``phases`` block.
    TypeError
        If a config key is unknown or the wrong type.
    """
    path = Path(path)
    with path.open("r") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    phases: dict[str, Any] = raw.pop("phases", {}) or {}
    merged = raw
    if phase is not None:
        if phase not in phases:
            raise KeyError(
                f"phase {phase!r} not found in {path} — available: {sorted(phases)}"
            )
        merged = _deep_merge(raw, phases[phase])

    top_level_kwargs: dict[str, Any] = {}
    for key, value in merged.items():
        if key in _NESTED_SECTIONS:
            top_level_kwargs[key] = _build_section(_NESTED_SECTIONS[key], value)
        else:
            top_level_kwargs[key] = value

    return TrainConfig(**top_level_kwargs)


def resolve_device(spec: str) -> torch.device:
    """Turn a ``"auto"|"cuda"|"cpu"`` string into a real ``torch.device``."""
    spec = spec.lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        return torch.device("cuda")
    if spec == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device spec: {spec!r} (expected auto|cuda|cpu)")


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Recursively turn a TrainConfig back into a plain dict (for logging)."""

    def _walk(obj: Any) -> Any:
        if is_dataclass(obj):
            return {f.name: _walk(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, (list, tuple)):
            return [_walk(x) for x in obj]
        return obj

    return _walk(cfg)  # type: ignore[return-value]
