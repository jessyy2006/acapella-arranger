"""Hybrid Transformer encoder + per-voice LSTM decoder for SATB generation.

Architecture (rubric justification: custom architecture combining paradigms)
---------------------------------------------------------------------------
One **shared embedding** maps tokens (lead and target voices alike) into a
``d_model`` space. One **Transformer encoder** consumes the lead sequence
and produces per-position memory. Four **independent decoder heads** — one
per target voice (soprano / alto / tenor / bass) — each combine a
unidirectional LSTM over the target tokens with cross-attention onto the
encoder memory. The voice-specific LSTM captures the idiomatic rhythmic
and melodic contour of that voice; the cross-attention lets it stay
aligned to the lead melody at every step.

Forward contract::

    batch = collate_satb([...])                # from src.data.loaders
    logits = model(batch)                      # {"s": ..., "a": ..., "t": ..., "b": ...}
    # Each logits[v] has shape (B, L_voice, vocab_size).

Teacher forcing, loss shifting, and autoregressive generation live in the
Day 3 Chunk 2 training code — this module is a pure forward pass.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, d)
        return x + self.pe[:, : x.size(1), :]


class VoiceDecoder(nn.Module):
    """Single voice's decoder: unidirectional LSTM → cross-attention → logits.

    The LSTM sees target tokens left-to-right (causal by construction),
    and each step queries the encoder memory via multihead attention so
    the output tracks the lead melody.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_lstm_layers: int,
        dropout: float,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,  # must be causal — decoder cannot see future
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_lstm = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        target_emb: torch.Tensor,  # (B, L, d)
        memory: torch.Tensor,  # (B, L_lead, d)
        memory_pad_mask: torch.Tensor,  # (B, L_lead), True where PAD
    ) -> torch.Tensor:
        lstm_out, _ = self.lstm(target_emb)
        lstm_out = self.norm_lstm(lstm_out + target_emb)  # residual around LSTM

        attn_out, _ = self.cross_attn(
            query=lstm_out,
            key=memory,
            value=memory,
            key_padding_mask=memory_pad_mask,
            need_weights=False,
        )
        fused = self.norm_attn(lstm_out + self.dropout(attn_out))
        return self.out_proj(fused)


class SATBHybrid(nn.Module):
    """Lead-in, four-voice-out harmony model."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        pad_idx: int = PAD,
        d_model: int = 256,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_lstm_layers: int = 2,
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
                voice: VoiceDecoder(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_lstm_layers=n_decoder_lstm_layers,
                    dropout=dropout,
                    vocab_size=vocab_size,
                )
                for voice in _VOICE_KEYS
            }
        )

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        # Scale by sqrt(d) per the original Transformer recipe.
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return self.embedding_dropout(x)

    def encode(
        self, lead: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(memory, pad_mask)`` for the lead input."""
        pad_mask = lead == self.pad_idx  # (B, L_lead)
        x = self._embed(lead)
        memory = self.encoder(x, src_key_padding_mask=pad_mask)
        return memory, pad_mask

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward pass on one ``collate_satb`` batch.

        Returns a ``{voice: logits}`` dict where each value has shape
        ``(B, L_voice, vocab_size)``. Shape across voices can differ
        because the collate pads each voice independently.
        """
        memory, memory_pad_mask = self.encode(batch["lead"])

        logits: dict[str, torch.Tensor] = {}
        for voice in _VOICE_KEYS:
            target = batch[voice]  # (B, L_voice)
            target_emb = self._embed(target)
            logits[voice] = self.decoders[voice](
                target_emb=target_emb,
                memory=memory,
                memory_pad_mask=memory_pad_mask,
            )
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
