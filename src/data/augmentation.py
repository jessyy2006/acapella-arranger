"""Training-time augmentation primitives for tokenised voice lines.

Two orthogonal transforms:

1. **Key transposition** — shift every PITCH token by *s* semitones.
   Specials, REST, and DURATION tokens are untouched. Any transposition
   that would push a note outside the MIDI ``[0, 127]`` range returns
   ``None`` so the caller can skip the augmentation silently. Typical
   use: sweep ``DEFAULT_TRANSPOSITION_RANGE`` to generate ×12 variants.

2. **Sliding-window chunking** — split a full-song sequence into
   overlapping ``window_bars``-measure segments with ``hop_bars`` stride,
   cut on ``BAR`` boundaries. Each window is re-wrapped with SOS/EOS so
   downstream code doesn't need to special-case position.

Both functions operate on ``list[int]`` token streams produced by
``src.data.tokenizer.encode_part``. They are deliberately stateless so
they compose cleanly with any caching / multiprocessing strategy.
"""

from __future__ import annotations

from typing import Final

from src.data.vocab import (
    BAR,
    EOS,
    NUM_PITCHES,
    PITCH_OFFSET,
    SOS,
    is_pitch_token,
)

# 12 transpositions covering every key; centred so the identity (0) is
# included and extremes are symmetric.
DEFAULT_TRANSPOSITION_RANGE: Final[tuple[int, ...]] = tuple(range(-5, 7))

DEFAULT_WINDOW_BARS: Final[int] = 8
DEFAULT_HOP_BARS: Final[int] = 4


def transpose_tokens(tokens: list[int], semitones: int) -> list[int] | None:
    """Return a new token list with every pitch shifted by ``semitones``.

    Returns ``None`` if any pitch token would land outside ``[0, 127]``.
    Non-pitch tokens are copied unchanged.
    """
    if semitones == 0:
        return list(tokens)

    out: list[int] = []
    for tok in tokens:
        if is_pitch_token(tok):
            new_midi = (tok - PITCH_OFFSET) + semitones
            if not 0 <= new_midi < NUM_PITCHES:
                return None
            out.append(PITCH_OFFSET + new_midi)
        else:
            out.append(tok)
    return out


def sliding_windows(
    tokens: list[int],
    *,
    window_bars: int = DEFAULT_WINDOW_BARS,
    hop_bars: int = DEFAULT_HOP_BARS,
) -> list[list[int]]:
    """Chunk a tokenised voice into overlapping fixed-length segments.

    The input is split on ``BAR`` tokens into measures; windows take
    ``window_bars`` consecutive measures with stride ``hop_bars``. Each
    returned segment is framed with ``SOS`` / ``EOS``. Songs shorter
    than ``window_bars`` are returned as a single shorter window.
    """
    if window_bars <= 0:
        raise ValueError(f"window_bars must be positive, got {window_bars}")
    if hop_bars <= 0:
        raise ValueError(f"hop_bars must be positive, got {hop_bars}")

    body = [t for t in tokens if t not in (SOS, EOS)]

    measures: list[list[int]] = [[]]
    for tok in body:
        if tok == BAR:
            measures.append([])
        else:
            measures[-1].append(tok)

    if sum(len(m) for m in measures) == 0:
        return []

    if len(measures) <= window_bars:
        return [_wrap_measures(measures)]

    windows: list[list[int]] = []
    start = 0
    while start < len(measures):
        end = start + window_bars
        chunk = measures[start:end]
        if not any(chunk):
            break
        windows.append(_wrap_measures(chunk))
        if end >= len(measures):
            break
        start += hop_bars
    return windows


def _wrap_measures(measures: list[list[int]]) -> list[int]:
    """Join measures with BAR separators and wrap with SOS/EOS."""
    out: list[int] = [SOS]
    for i, measure in enumerate(measures):
        if i > 0:
            out.append(BAR)
        out.extend(measure)
    out.append(EOS)
    return out
