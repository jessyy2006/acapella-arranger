"""Per-voice cross-entropy loss + token accuracy for SATB training.

The model emits one ``(B, L, V)`` logit tensor per voice. Teacher forcing
shifts by one step: logit at position ``t`` predicts target token at
``t+1``, so we pair ``logits[:, :-1]`` with ``targets[:, 1:]`` and mask
out PAD positions via ``ignore_index``.

Per-voice losses are returned alongside the summed total so the training
CSV can show which voice is (mis)behaving — catching e.g. bass dominating
or a voice collapsing to REST.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor

_TARGET_VOICES: Final[tuple[str, ...]] = ("s", "a", "t", "b")


def compute_loss(
    logits: dict[str, Tensor],
    batch: dict[str, Tensor],
    pad_idx: int,
) -> tuple[Tensor, dict[str, float]]:
    """Return ``(total_loss, {voice: per-voice_ce_float})``.

    The total loss is the **sum** across the four voice CEs, not the mean —
    the training spec asks for summed per-voice CE so gradient magnitude
    reflects the four-voice objective.
    """
    per_voice: dict[str, float] = {}
    total: Tensor | None = None

    for voice in _TARGET_VOICES:
        voice_logits = logits[voice]  # (B, L, V)
        targets = batch[voice]  # (B, L)
        # Shift: predict token t+1 given tokens 0..t.
        pred = voice_logits[:, :-1, :].reshape(-1, voice_logits.size(-1))
        gold = targets[:, 1:].reshape(-1)
        loss_v = F.cross_entropy(pred, gold, ignore_index=pad_idx)
        per_voice[voice] = loss_v.detach().item()
        total = loss_v if total is None else total + loss_v

    assert total is not None  # _TARGET_VOICES is non-empty
    return total, per_voice


def compute_accuracy(
    logits: dict[str, Tensor],
    batch: dict[str, Tensor],
    pad_idx: int,
) -> dict[str, float]:
    """Per-voice top-1 token accuracy on non-PAD positions."""
    out: dict[str, float] = {}
    for voice in _TARGET_VOICES:
        voice_logits = logits[voice]
        targets = batch[voice]
        pred = voice_logits[:, :-1, :].argmax(dim=-1)  # (B, L-1)
        gold = targets[:, 1:]  # (B, L-1)
        mask = gold != pad_idx
        correct = ((pred == gold) & mask).sum().item()
        total = mask.sum().item()
        out[voice] = correct / total if total > 0 else 0.0
    return out
