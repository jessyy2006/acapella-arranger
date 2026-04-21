"""Evaluation metrics for SATB arrangement models.

Metrics assume *teacher forcing*: ``logits[:, t]`` predicts ``targets[:, t+1]``.
The accuracy helpers always compare ``logits[:, :-1].argmax(-1)`` against
``targets[:, 1:]`` so bar boundaries and padding positions line up.

Two input shapes show up in this module:

- Per-token metrics (``next_token_accuracy``, ``per_voice_accuracy``,
  ``duration_bucket_accuracy``, ``bar_accuracy``) take per-voice logits of
  shape ``(B, L, V)`` and targets of shape ``(B, L)``.
- List-based metrics (``pitch_range_compliance``, ``voice_crossing_rate``)
  take per-sample token sequences as ``dict[str, list[list[int]]]`` —
  outer list is batch, inner is the token stream for that sample. Callers
  must NOT flatten the batch before calling; flattening mixes samples
  together and makes the voice-crossing timeline reconstruction meaningless.
"""

from __future__ import annotations

from typing import Final

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


def pitch_range_compliance(
    pred_tokens_by_voice: dict[str, list[list[int]]],
) -> dict[str, float]:
    """Fraction of predicted pitch tokens inside each voice's training range.

    Micro-averaged across samples. If a voice never predicts any pitch
    token (denominator 0), that voice returns ``nan`` rather than a
    silent 1.0. Pair with ``pitch_count_by_voice`` in reports so a 1.0
    with only a handful of pitches is auditable.
    """
    out: dict[str, float] = {}
    for v in _VOICES:
        lo, hi = _VOICE_RANGES[v]
        in_range = 0
        total = 0
        for sample in pred_tokens_by_voice[v]:
            for tok in sample:
                pitch = token_to_pitch(int(tok))
                if pitch is None:
                    continue
                total += 1
                if lo <= pitch <= hi:
                    in_range += 1
        out[v] = float(in_range / total) if total > 0 else float("nan")
    return out


def pitch_count_by_voice(
    pred_tokens_by_voice: dict[str, list[list[int]]],
) -> dict[str, int]:
    """Denominator of ``pitch_range_compliance`` per voice — how many pitch
    tokens the model actually emitted. Reports should surface this alongside
    the compliance fraction so a 1.0 with count=3 isn't mistaken for signal.
    """
    return {
        v: sum(
            1
            for sample in pred_tokens_by_voice[v]
            for tok in sample
            if token_to_pitch(int(tok)) is not None
        )
        for v in _VOICES
    }


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
    """Fraction of bars where all four voices match exactly.

    Uses the same teacher-forcing shift as ``next_token_accuracy`` so pred
    and gold cover the same positions (``1..L`` of targets, ``0..L-1`` of
    logits). Without the shift, pred and gold BAR boundaries misalign by
    one token and no bar can ever match — which is the bug that made this
    metric return 0.0000 on every evaluated checkpoint.
    """
    batch_size = int(next(iter(targets_by_voice.values())).shape[0])
    good = 0
    total = 0
    for b in range(batch_size):
        pred = {v: logits_by_voice[v][b, :-1, :].argmax(dim=-1).tolist() for v in _VOICES}
        tgt = {v: targets_by_voice[v][b, 1:].tolist() for v in _VOICES}
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


def voice_crossing_rate(
    pred_tokens_by_voice: dict[str, list[list[int]]],
) -> float:
    """Fraction of aligned time-steps where SATB ordering is violated.

    Micro-averaged across samples. Each sample gets its own reconstructed
    timeline (pitch-vs-time per voice); we count every time-step where all
    four voices have a pitch and ``S >= A >= T >= B`` does not hold, then
    divide one aggregate violation count by one aggregate considered
    count. The prior implementation took a flattened batch and walked it
    as a single stream — which silently joined the last notes of sample
    ``i`` to the first notes of sample ``i+1`` and produced meaningless
    crossings at batch boundaries.
    """
    n_samples = len(pred_tokens_by_voice["s"])
    violations_total = 0
    considered_total = 0
    for i in range(n_samples):
        timelines = {v: _tokens_to_timeline(pred_tokens_by_voice[v][i]) for v in _VOICES}
        n = min(len(timelines[v]) for v in _VOICES)
        for step in range(n):
            s, a, t, b = (
                timelines["s"][step],
                timelines["a"][step],
                timelines["t"][step],
                timelines["b"][step],
            )
            if any(x is None for x in (s, a, t, b)):
                continue
            considered_total += 1
            assert s is not None and a is not None and t is not None and b is not None
            if not (s >= a >= t >= b):
                violations_total += 1
    if considered_total == 0:
        return float("nan")
    return float(violations_total / considered_total)
