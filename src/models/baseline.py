"""Pure-Transformer baseline SATB model (no LSTM) — ablation comparison.

This module provides :class:`SATBBaseline`, a drop-in alternative to
:class:`src.models.hybrid.SATBHybrid` for the rubric-required ablation. It
mirrors the hybrid's topology *exactly* except for the decoder self-attention
mechanism, so any measured gap attributes cleanly to "LSTM vs Transformer
self-attention" rather than to incidental differences in parameter sharing.

Design choices
----------------------------------------------
- **Four separate decoder stacks**, one per voice (soprano/alto/tenor/bass),
  mirroring the hybrid's four independent ``VoiceDecoder`` heads. This is the
  spec-recommended design in ``docs/specs/baseline_model.md``: sharing the
  decoder across voices would confound the ablation with a second axis of
  variation (shared-vs-separate) on top of the one we care about
  (LSTM-vs-Transformer).
- **Shared token embedding + sinusoidal positional encoding** across the lead
  encoder and all four target decoders — same choice the hybrid makes, kept
  identical so the lead-side comparison is apples-to-apples.
- **No input/output embedding tying**: the output projections are per-voice
  ``nn.Linear`` heads, which matches how the hybrid projects to vocab.
- **Pre-norm layers (`norm_first=True`)** and **embedding scaling by
  `sqrt(d_model)`** — both match the hybrid's stability recipe.
- **Default `n_decoder_layers=2`** keeps total parameter count within ±50%
  of the hybrid. Four 3-layer stacks would nearly double the hybrid's budget
  and confound the ablation on model size.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import Tensor, nn

from src.data.vocab import PAD, VOCAB_SIZE

_VOICE_KEYS: Final[tuple[str, ...]] = ("s", "a", "t", "b")


class SinusoidalPositionalEncoding(nn.Module):
    """Classic Transformer positional encoding, registered as a buffer."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:  # (B, L, d)
        return x + self.pe[:, : x.size(1), :]


class VoiceTransformerDecoder(nn.Module):
    """Single voice's Transformer decoder stack + output projection.

    Causal self-attention (via ``generate_square_subsequent_mask``) makes the
    block autoregressive; cross-attention onto the encoder memory keeps the
    output aligned to the lead melody. Mirrors the role of
    :class:`src.models.hybrid.VoiceDecoder` but with Transformer self-attn
    in place of the LSTM.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        vocab_size: int,
    ) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        target_emb: Tensor,  # (B, L_tgt, d)
        memory: Tensor,  # (B, L_lead, d)
        tgt_pad_mask: Tensor,  # (B, L_tgt), True where PAD
        memory_pad_mask: Tensor,  # (B, L_lead), True where PAD
    ) -> Tensor:
        l_tgt = target_emb.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(l_tgt).to(
            device=target_emb.device
        )
        decoded = self.decoder(
            tgt=target_emb,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        return self.out_proj(decoded)


class SATBBaseline(nn.Module):
    """Lead-in, four-voice-out harmony model — pure Transformer (no LSTM)."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        pad_idx: int = PAD,
        d_model: int = 256,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoder_layers
        )

        self.decoders = nn.ModuleDict(
            {
                voice: VoiceTransformerDecoder(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_decoder_layers,
                    d_ff=d_ff,
                    dropout=dropout,
                    vocab_size=vocab_size,
                )
                for voice in _VOICE_KEYS
            }
        )

    def _embed(self, tokens: Tensor) -> Tensor:
        # Scale by sqrt(d) per the original Transformer recipe.
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return self.embedding_dropout(x)

    def encode(self, lead: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(memory, pad_mask)`` for the lead input."""
        pad_mask = lead == self.pad_idx  # (B, L_lead)
        x = self._embed(lead)
        memory = self.encoder(x, src_key_padding_mask=pad_mask)
        return memory, pad_mask

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass on one ``collate_satb`` batch.

        Returns a ``{voice: logits}`` dict where each value has shape
        ``(B, L_voice, vocab_size)``.
        """
        memory, memory_pad_mask = self.encode(batch["lead"])

        logits: dict[str, Tensor] = {}
        for voice in _VOICE_KEYS:
            target = batch[voice]  # (B, L_voice)
            target_emb = self._embed(target)
            tgt_pad_mask = target == self.pad_idx
            logits[voice] = self.decoders[voice](
                target_emb=target_emb,
                memory=memory,
                tgt_pad_mask=tgt_pad_mask,
                memory_pad_mask=memory_pad_mask,
            )
        return logits

    def num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
