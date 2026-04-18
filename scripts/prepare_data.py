"""Preprocess raw scores into batched train/val/test datasets on disk.

Loads JSB Chorales (via music21.corpus) and jaCappella scores (from the
local ``data/raw/jacappella/`` tree), runs the shared tokenizer +
sliding-window + ×12 transposition augmentation pipeline, and writes
three ``torch.save``'d ``SATBDataset`` objects:

    data/processed/train.pt
    data/processed/val.pt
    data/processed/test.pt

Consumers should load these files with
:func:`src.data.loaders.load_dataset` rather than calling ``torch.load``
directly — PyTorch's ``weights_only=True`` default (2.6+) refuses to
unpickle the full ``SATBDataset`` object.

Idempotent — existing output files are kept unless ``--force`` is
passed. All knobs (seed, split ratios, window size, per-source limits)
are exposed as CLI flags so the same script can drive both smoke runs
and full preprocessing.

Usage:
    python scripts/prepare_data.py                  # full build
    python scripts/prepare_data.py --force          # rebuild from scratch
    python scripts/prepare_data.py --limit-jsb 5 \\
        --limit-jacappella 2 --force                # quick smoke
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow ``python scripts/prepare_data.py`` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from music21 import corpus, stream

from src.data.load import is_clean_satb, parse_lenient
from src.data.loaders import build_split

_DEFAULT_OUT = PROJECT_ROOT / "data" / "processed"
_DEFAULT_JACAPPELLA_ROOT = PROJECT_ROOT / "data" / "raw" / "jacappella"


@dataclass
class PrepareConfig:
    out_dir: Path = _DEFAULT_OUT
    jacappella_root: Path = _DEFAULT_JACAPPELLA_ROOT
    seed: int = 42
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    window_bars: int = 8
    hop_bars: int = 4
    limit_jsb: int | None = None
    limit_jacappella: int | None = None
    force: bool = False


def _load_jsb(limit: int | None) -> list[stream.Score]:
    print("Loading JSB chorales from music21 corpus...")
    out: list[stream.Score] = []
    for score in corpus.chorales.Iterator():
        if not is_clean_satb(score):
            continue
        out.append(score)
        if limit is not None and len(out) >= limit:
            break
    print(f"  -> {len(out)} clean SATB chorales")
    return out


# jaCappella ships each song as three MusicXML variants (base, romaji,
# SVS) that share identical pitches and rhythm and differ only in
# lyrics / phoneme encoding. Since the tokenizer is lyric-blind, loading
# all three would triple-count each song and leak duplicates across the
# train / val / test split. We keep the un-suffixed canonical file only.
_JACAPPELLA_SKIP_SUFFIXES: tuple[str, ...] = ("_romaji", "_SVS")


def _is_canonical_jacappella(path: Path) -> bool:
    stem = path.stem
    return not any(stem.endswith(suffix) for suffix in _JACAPPELLA_SKIP_SUFFIXES)


def _load_jacappella(root: Path, limit: int | None) -> list[stream.Score]:
    if not root.exists():
        print(f"jaCappella directory missing: {root}")
        print("  (skipping — run scripts/download_data.py to populate)")
        return []
    print(f"Loading jaCappella scores from {root}...")
    xmls = [p for p in sorted(root.rglob("*.musicxml")) if _is_canonical_jacappella(p)]
    out: list[stream.Score] = []
    for path in xmls:
        try:
            out.append(parse_lenient(path))
        except Exception as exc:  # noqa: BLE001 — user-visible diagnostic
            print(f"  ! failed to parse {path.relative_to(root)}: {exc}")
            continue
        if limit is not None and len(out) >= limit:
            break
    print(f"  -> {len(out)} jaCappella scores (canonical variant only)")
    return out


def run(config: PrepareConfig) -> int:
    """Execute the full preprocessing pipeline. Returns a shell-style exit code."""
    targets = {
        name: config.out_dir / f"{name}.pt" for name in ("train", "val", "test")
    }

    if not config.force and all(p.exists() for p in targets.values()):
        print("All outputs already exist; pass --force to rebuild:")
        for name, path in targets.items():
            size_mb = path.stat().st_size / 1e6
            print(f"  {name}: {path} ({size_mb:.1f} MB)")
        return 0

    config.out_dir.mkdir(parents=True, exist_ok=True)

    jsb = _load_jsb(config.limit_jsb)
    jac = _load_jacappella(config.jacappella_root, config.limit_jacappella)

    if not jsb and not jac:
        print("\nERROR: no input scores found.", file=sys.stderr)
        print("Run `python scripts/download_data.py` first.", file=sys.stderr)
        return 1

    print(
        f"\nBuilding splits (seed={config.seed}, ratios={config.ratios}, "
        f"window={config.window_bars} bars, hop={config.hop_bars} bars)..."
    )
    splits = build_split(
        jsb,
        jac,
        ratios=config.ratios,
        seed=config.seed,
        window_bars=config.window_bars,
        hop_bars=config.hop_bars,
    )

    print("\nWriting datasets:")
    for name, dataset in splits.items():
        path = targets[name]
        torch.save(dataset, path)
        size_mb = path.stat().st_size / 1e6
        display = (
            path.relative_to(PROJECT_ROOT)
            if path.is_relative_to(PROJECT_ROOT)
            else path
        )
        print(
            f"  {name}: {len(dataset):>6} examples from "
            f"{dataset.songs_kept:>3} songs ({dataset.songs_skipped} skipped)  "
            f"-> {display} ({size_mb:.1f} MB)"
        )

    print("\nDone.")
    return 0


def _parse_args(argv: list[str] | None = None) -> PrepareConfig:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument(
        "--jacappella-root", type=Path, default=_DEFAULT_JACAPPELLA_ROOT
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument("--window-bars", type=int, default=8)
    parser.add_argument("--hop-bars", type=int, default=4)
    parser.add_argument("--limit-jsb", type=int, default=None)
    parser.add_argument("--limit-jacappella", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    ns = parser.parse_args(argv)

    return PrepareConfig(
        out_dir=ns.out,
        jacappella_root=ns.jacappella_root,
        seed=ns.seed,
        ratios=tuple(ns.ratios),  # type: ignore[arg-type]
        window_bars=ns.window_bars,
        hop_bars=ns.hop_bars,
        limit_jsb=ns.limit_jsb,
        limit_jacappella=ns.limit_jacappella,
        force=ns.force,
    )


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
