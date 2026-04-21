"""Evaluation metrics for SATB arrangement models.

Metrics assume *teacher forcing*: ``logits[:, t]`` predicts ``targets[:, t+1]``.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import torch
from torch import Tensor

from src.data.vocab import BAR, PAD, REST, is_duration_token, token_to_duration, token_to_pitch

_VOICES: Final[tuple[str, ...]] = ("s", "a", "t", "b")
_VOICE_RANGES: Final[dict[str, tuple[int, int]]] = {
    "s": (55, 81),
    "a": (50, 76),
    "t": (47, 74),
    "b": (31, 64),
}


def next_token_accuracy(logits: Tensor, targets: Tensor, pad_idx: int = PAD) -> float:
    """Fraction of non-PAD next-token positions predicted correctly."""
    pred = logits[:, :-1, :].argmax(dim=-1)
    gold = targets[:, 1:]
    mask = gold != pad_idx
    denom = int(mask.sum().item())
    if denom == 0:
        return float("nan")
    correct = ((pred == gold) & mask).sum().item()
    return float(correct / denom)


def per_voice_accuracy(
    logits_by_voice: dict[str, Tensor], targets_by_voice: dict[str, Tensor], pad_idx: int = PAD
) -> dict[str, float]:
    return {v: next_token_accuracy(logits_by_voice[v], targets_by_voice[v], pad_idx) for v in _VOICES}


def duration_bucket_accuracy(logits_by_voice: dict[str, Tensor], targets_by_voice: dict[str, Tensor]) -> float:
    """Next-token accuracy restricted to duration tokens (in gold stream)."""
    num = 0
    den = 0
    for v in _VOICES:
        logits = logits_by_voice[v]
        targets = targets_by_voice[v]
        pred = logits[:, :-1, :].argmax(dim=-1)
        gold = targets[:, 1:]
        gold_list = gold.detach().cpu().tolist()
        dur_mask = torch.tensor(
            [[is_duration_token(int(t)) and int(t) != PAD for t in row] for row in gold_list],
            device=gold.device,
            dtype=torch.bool,
        )
        den += int(dur_mask.sum().item())
        num += int(((pred == gold) & dur_mask).sum().item())
    return float(num / den) if den > 0 else float("nan")


def pitch_range_compliance(pred_tokens_by_voice: dict[str, list[int]]) -> dict[str, float]:
    """Fraction of pitch tokens within observed training ranges."""
    out: dict[str, float] = {}
    for v in _VOICES:
        lo, hi = _VOICE_RANGES[v]
        midi = [p for p in (token_to_pitch(t) for t in pred_tokens_by_voice[v]) if p is not None]
        if not midi:
            out[v] = float("nan")
            continue
        out[v] = float(sum(1 for p in midi if lo <= p <= hi) / len(midi))
    return out


def _split_by_bar(tokens: list[int]) -> list[list[int]]:
    cleaned = [t for t in tokens if t != PAD]
    out: list[list[int]] = [[]]
    for t in cleaned:
        if t == BAR:
            out.append([])
        else:
            out[-1].append(t)
    return [b for b in out if b]


def bar_accuracy(logits_by_voice: dict[str, Tensor], targets_by_voice: dict[str, Tensor]) -> float:
    """Fraction of bars where all four voices match exactly."""
    batch_size = int(next(iter(targets_by_voice.values())).shape[0])
    good = 0
    total = 0
    for b in range(batch_size):
        pred = {v: logits_by_voice[v][b].argmax(dim=-1).tolist() for v in _VOICES}
        tgt = {v: targets_by_voice[v][b].tolist() for v in _VOICES}
        pred_bars = {v: _split_by_bar(pred[v]) for v in _VOICES}
        tgt_bars = {v: _split_by_bar(tgt[v]) for v in _VOICES}
        n = min(min(len(pred_bars[v]), len(tgt_bars[v])) for v in _VOICES)
        for i in range(n):
            total += 1
            if all(pred_bars[v][i] == tgt_bars[v][i] for v in _VOICES):
                good += 1
    return float(good / total) if total > 0 else float("nan")


def _tokens_to_timeline(tokens: list[int]) -> list[int | None]:
    out: list[int | None] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in (PAD, BAR):
            i += 1
            continue
        if t == REST or token_to_pitch(t) is not None:
            if i + 1 >= len(tokens):
                break
            dur = token_to_duration(tokens[i + 1])
            if dur is None:
                i += 1
                continue
            pitch = None if t == REST else token_to_pitch(t)
            out.extend([pitch] * int(dur))
            i += 2
            continue
        i += 1
    return out


def voice_crossing_rate(pred_tokens_by_voice: dict[str, list[int]]) -> float:
    timelines = {v: _tokens_to_timeline(pred_tokens_by_voice[v]) for v in _VOICES}
    n = min(len(timelines[v]) for v in _VOICES)
    if n == 0:
        return float("nan")
    violations = 0
    considered = 0
    for i in range(n):
        s, a, t, b = (timelines["s"][i], timelines["a"][i], timelines["t"][i], timelines["b"][i])
        if any(x is None for x in (s, a, t, b)):
            continue
        considered += 1
        assert s is not None and a is not None and t is not None and b is not None
        if not (s >= a >= t >= b):
            violations += 1
    return float(violations / considered) if considered > 0 else float("nan")

