"""Train/val/test split (by song) and DataLoader factory for SATBDataset.

``split_scores_by_song`` shuffles and partitions a flat list of scores;
``build_split`` wraps that into per-source splits and wraps each shard
in a :class:`src.data.dataset.SATBDataset`. By construction, no song
ever appears in more than one split — leakage would inflate validation
scores and lead to a silent overfit.

``collate_satb`` pads variable-length sequences per voice and emits the
``{voice}_len`` length tensor needed by the attention mask in the
downstream Transformer+LSTM hybrid.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from music21 import stream
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.data.dataset import SATBDataset, Source, VOICE_KEYS
from src.data.vocab import PAD


def split_scores_by_song(
    scores: list[stream.Score],
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> tuple[list[stream.Score], list[stream.Score], list[stream.Score]]:
    """Partition a list of scores into (train, val, test) by whole song.

    Uses ``floor`` for train/val counts so test picks up any remainder —
    this avoids silently losing songs when ``len(scores) * ratios`` is
    not integer.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    shuffled = list(scores)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def build_split(
    jsb_scores: list[stream.Score],
    jacappella_scores: list[stream.Score],
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    augment_train: bool = True,
    **dataset_kwargs: Any,
) -> dict[str, SATBDataset]:
    """Build train/val/test SATBDatasets with per-source by-song splitting.

    Train gets augmentation (×12 transpositions); val and test are
    evaluated on non-augmented sequences so metrics reflect real data.
    ``dataset_kwargs`` are forwarded to :class:`SATBDataset` (e.g. to
    override ``window_bars`` or ``hop_bars``).
    """
    rng = random.Random(seed)

    jsb_train, jsb_val, jsb_test = split_scores_by_song(jsb_scores, ratios, rng)
    jac_train, jac_val, jac_test = split_scores_by_song(
        jacappella_scores, ratios, rng
    )

    def _label(scores: list[stream.Score], src: Source) -> list[tuple[Source, stream.Score]]:
        return [(src, s) for s in scores]

    return {
        "train": SATBDataset(
            _label(jsb_train, "jsb") + _label(jac_train, "jacappella"),
            augment=augment_train,
            **dataset_kwargs,
        ),
        "val": SATBDataset(
            _label(jsb_val, "jsb") + _label(jac_val, "jacappella"),
            augment=False,
            **dataset_kwargs,
        ),
        "test": SATBDataset(
            _label(jsb_test, "jsb") + _label(jac_test, "jacappella"),
            augment=False,
            **dataset_kwargs,
        ),
    }


def collate_satb(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad each voice independently; emit ``{voice}_len`` for masking."""
    out: dict[str, torch.Tensor] = {}
    for voice in VOICE_KEYS:
        seqs = [item[voice] for item in batch]
        out[voice] = pad_sequence(seqs, batch_first=True, padding_value=PAD)
        out[f"{voice}_len"] = torch.tensor(
            [len(s) for s in seqs], dtype=torch.long
        )
    return out


def make_dataloaders(
    splits: dict[str, SATBDataset],
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    """Wrap each split in a DataLoader with the padding collate."""
    loaders: dict[str, DataLoader] = {}
    for name, dataset in splits.items():
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train" and shuffle_train),
            num_workers=num_workers,
            collate_fn=collate_satb,
        )
    return loaders


def load_dataset(path: str | Path) -> SATBDataset:
    """Load a cached SATBDataset written by ``scripts/prepare_data.py``.

    Use this in place of ``torch.load(path)``. The cached payload is a
    pickled Python object (lists of ints, not tensor state), and
    PyTorch ≥2.6 defaults ``torch.load`` to ``weights_only=True``, which
    refuses to unpickle arbitrary Python objects. This helper passes
    ``weights_only=False`` explicitly.

    Only call this on ``.pt`` files you produced yourself — unpickling
    an untrusted payload can execute arbitrary code.
    """
    return torch.load(str(path), weights_only=False)
