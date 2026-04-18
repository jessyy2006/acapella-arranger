"""PyTorch Dataset for ``(lead, soprano, alto, tenor, bass)`` training tuples.

The dataset pre-materialises all examples at construction time:

    for each song:
        tokenise every voice
        slide a (window_bars, hop_bars) window across the voices
        for each window, apply every requested transposition
        keep only windows where all five voices survive the shift

Training examples are therefore fully in-memory ``dict[str, list[int]]``
records; ``__getitem__`` converts the requested record to ``LongTensor``s
on demand so the memory footprint stays flat across worker processes.

Per-source voice routing (see Day 1 plan):

* **JSB** — soprano doubles as the lead input. Five tensors per example
  but ``lead == s`` token-for-token. Phase A pretraining teaches the
  model the "melody-in-S" convention.
* **jaCappella** — ``lead_vocal`` feeds ``lead``; ``soprano``/``alto``/
  ``tenor``/``bass`` feed the four targets. Phase B fine-tuning teaches
  the model to diverge the generated soprano from the lead melody.
"""

from __future__ import annotations

import logging
from typing import Final, Literal

import torch
from music21 import stream
from torch.utils.data import Dataset

from src.data.augmentation import (
    DEFAULT_HOP_BARS,
    DEFAULT_TRANSPOSITION_RANGE,
    DEFAULT_WINDOW_BARS,
    sliding_windows,
    transpose_tokens,
)
from src.data.tokenizer import encode_part

logger = logging.getLogger(__name__)

Source = Literal["jsb", "jacappella"]

VOICE_KEYS: Final[tuple[str, ...]] = ("lead", "s", "a", "t", "b")

_JSB_PART_NAMES: Final[dict[str, str]] = {
    "lead": "Soprano",
    "s": "Soprano",
    "a": "Alto",
    "t": "Tenor",
    "b": "Bass",
}

_JACAPPELLA_PART_NAMES: Final[dict[str, str]] = {
    "lead": "lead_vocal",
    "s": "soprano",
    "a": "alto",
    "t": "tenor",
    "b": "bass",
}


def _find_part(score: stream.Score, name: str) -> stream.Part | None:
    for p in score.parts:
        if (p.partName or "").strip() == name:
            return p
    return None


def extract_voices_jsb(score: stream.Score) -> dict[str, stream.Part | None]:
    """Map voice keys to music21 Parts for a JSB chorale (soprano doubles as lead)."""
    return {k: _find_part(score, _JSB_PART_NAMES[k]) for k in VOICE_KEYS}


def extract_voices_jacappella(score: stream.Score) -> dict[str, stream.Part | None]:
    """Map voice keys to music21 Parts for a jaCappella song."""
    return {k: _find_part(score, _JACAPPELLA_PART_NAMES[k]) for k in VOICE_KEYS}


_EXTRACTORS = {
    "jsb": extract_voices_jsb,
    "jacappella": extract_voices_jacappella,
}


class SATBDataset(Dataset):
    """Pre-materialised (lead, s, a, t, b) token-tuple dataset.

    Parameters
    ----------
    songs
        Iterable of ``(source, score)`` pairs where ``source`` is
        ``"jsb"`` or ``"jacappella"``.
    transpositions
        Semitone shifts to apply per window. Each shift that leaves any
        voice outside MIDI ``[0, 127]`` is silently skipped. Default is
        the full ×12 range; pass ``(0,)`` to disable transposition.
    window_bars, hop_bars
        Sliding-window parameters forwarded to
        :func:`src.data.augmentation.sliding_windows`.
    augment
        If ``False`` overrides ``transpositions`` to ``(0,)`` — used for
        validation and test splits.
    """

    def __init__(
        self,
        songs: list[tuple[Source, stream.Score]],
        *,
        transpositions: tuple[int, ...] = DEFAULT_TRANSPOSITION_RANGE,
        window_bars: int = DEFAULT_WINDOW_BARS,
        hop_bars: int = DEFAULT_HOP_BARS,
        augment: bool = True,
    ) -> None:
        shifts = tuple(transpositions) if augment else (0,)

        self._examples: list[dict[str, list[int]]] = []
        self._songs_kept = 0
        self._songs_skipped = 0

        for source, score in songs:
            if source not in _EXTRACTORS:
                raise ValueError(f"unknown source: {source!r}")
            voices = _EXTRACTORS[source](score)

            missing = [k for k, v in voices.items() if v is None]
            if missing:
                logger.warning(
                    "skipping %s song (missing parts: %s)", source, missing
                )
                self._songs_skipped += 1
                continue

            token_streams = {
                k: encode_part(v) for k, v in voices.items()  # type: ignore[arg-type]
            }
            windowed = {
                k: sliding_windows(
                    toks, window_bars=window_bars, hop_bars=hop_bars
                )
                for k, toks in token_streams.items()
            }

            # Voices sung together should produce the same number of
            # windows, but defensively take the minimum so a stray
            # pickup measure never mis-aligns the targets.
            n_windows = min(len(w) for w in windowed.values())
            if n_windows == 0:
                self._songs_skipped += 1
                continue
            self._songs_kept += 1

            for i in range(n_windows):
                base = {k: windowed[k][i] for k in VOICE_KEYS}
                for semitones in shifts:
                    shifted = {
                        k: transpose_tokens(seg, semitones)
                        for k, seg in base.items()
                    }
                    if any(v is None for v in shifted.values()):
                        continue
                    self._examples.append(shifted)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self._examples[idx]
        return {k: torch.tensor(ex[k], dtype=torch.long) for k in VOICE_KEYS}

    @property
    def songs_kept(self) -> int:
        return self._songs_kept

    @property
    def songs_skipped(self) -> int:
        return self._songs_skipped
