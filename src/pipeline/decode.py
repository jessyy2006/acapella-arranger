"""Autoregressive decoding for trained SATB models.

Exposes ``generate_voice_tokens`` — the greedy/sampled decoder used by
both ``scripts/sample_midi.py`` (the listening-loop CLI) and
``src/pipeline/run_pipeline.py`` (the end-to-end inference pipeline).
Keeping it in one module guarantees both callers use the same sampling
conventions and avoids drift between the two.

The decoder interleaves pitch and duration tokens per the tokenizer
grammar (``src/data/vocab.py``), and supports split temperatures so the
caller can sample pitch positions conservatively while letting duration
positions explore — the model's duration accuracy is high enough that a
single low temperature collapses every note onto the modal quarter-note
bucket and produces monotonous rhythm.
"""

from __future__ import annotations

from typing import Final

import torch

from src.data.vocab import EOS, PAD, REST, SOS, is_pitch_token


VOICES: Final[tuple[str, ...]] = ("s", "a", "t", "b")


@torch.no_grad()
def generate_voice_tokens(
    model: torch.nn.Module,
    lead: torch.Tensor,  # (1, L_lead) on device
    voice: str,
    max_len: int,
    device: torch.device,
    *,
    temperature: float = 0.5,
    duration_temperature: float | None = 1.1,
    top_k: int | None = 10,
    seed: int = 0,
) -> list[int]:
    """Autoregressive decode of a single voice conditioned on ``lead``.

    Parameters
    ----------
    model
        A trained ``SATBHybrid`` or ``SATBBaseline`` instance in eval mode
        on ``device``.
    lead
        The lead melody tokens as a ``(1, L_lead)`` long tensor.
    voice
        One of ``"s" | "a" | "t" | "b"``. Determines which decoder head's
        logits we read and feed back in.
    max_len
        Hard cap on emitted tokens (SOS counts toward the cap). Protects
        against runaway generation when the model never predicts EOS.
    temperature
        Sampling temperature for pitch / BAR / EOS positions. ``0.0`` is
        greedy (argmax). Values > 0 scale logits before softmax.
    duration_temperature
        Override for duration-token positions (positions following a pitch
        or REST). Raising this above ``temperature`` keeps note lengths
        varied; ``None`` falls back to ``temperature``.
    top_k
        Clip to the top-k most likely tokens before sampling. Keeps
        stochastic samples on-distribution. ``None`` or ``0`` disables.
    seed
        Seed for the CPU generator used by ``torch.multinomial``. Same
        seed → deterministic sample.

    Returns
    -------
    A list of token ids starting with ``SOS``, terminating at ``EOS``,
    ``PAD``, or ``max_len`` (whichever comes first).
    """
    tokens: list[int] = [SOS]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dur_temp = duration_temperature if duration_temperature is not None else temperature

    for _ in range(max_len):
        current = torch.tensor([tokens], dtype=torch.long, device=device)
        batch: dict[str, torch.Tensor] = {
            "lead": lead,
            "lead_len": torch.tensor([lead.shape[1]], device=device),
        }
        # Every voice key must be present for the model to forward; we
        # only *read* the voice we're decoding.
        for v in VOICES:
            batch[v] = current if v == voice else torch.tensor([[SOS]], device=device)
            batch[f"{v}_len"] = torch.tensor([1], device=device)

        logits = model(batch)[voice][0, -1, :]  # (V,)

        # Grammar: (pitch|REST, duration) pairs. If the *previous* emitted
        # token was a pitch or REST, the next one is structurally a
        # duration — use the duration temperature there. This keeps the
        # pitch stream conservative while letting rhythm breathe.
        prev_tok = tokens[-1]
        expecting_duration = is_pitch_token(prev_tok) or prev_tok == REST
        effective_temp = dur_temp if expecting_duration else temperature

        if effective_temp <= 0.0:
            next_tok = int(logits.argmax(dim=-1).item())
        else:
            scaled = logits / effective_temp
            if top_k is not None and top_k > 0:
                kth = torch.topk(scaled, k=top_k).values[-1]
                scaled = torch.where(
                    scaled < kth, torch.full_like(scaled, float("-inf")), scaled
                )
            probs = torch.softmax(scaled, dim=-1).cpu()
            next_tok = int(
                torch.multinomial(probs, num_samples=1, generator=generator).item()
            )

        if next_tok == PAD:
            break
        tokens.append(next_tok)
        if next_tok == EOS:
            break

    return tokens
