"""Pure-Transformer baseline SATB model (no LSTM).

This module provides :class:`SATBBaseline`, a drop-in alternative to
:class:`src.models.hybrid.SATBHybrid` for ablation comparisons. The encoder
consumes the lead sequence; each of the four target voices uses its own
Transformer decoder stack with **causal self-attention** and cross-attention
to the shared encoder memory.

Design choices
----------------------------------------------
- Shared token embedding + sinusoidal positional encoding for all streams.
- One shared Transformer decoder stack, with a small per-voice output head.
  This keeps the baseline parameter count comparable to the hybrid so the
  ablation is not confounded by model size.
- Pre-norm Transformer layers (``norm_first=True``) and embedding scaling by
  ``sqrt(d_model)`` to match the hybrid's stability choices.
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
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:  # (B, L, d)
        return x + self.pe[:, : x.size(1), :]


class SATBBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        pad_idx: int = PAD,
        d_model: int = 256,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        self.out_proj = nn.ModuleDict(
            {voice: nn.Linear(d_model, vocab_size) for voice in _VOICE_KEYS}
        )

    def _embed(self, tokens: Tensor) -> Tensor:
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return self.embedding_dropout(x)

    def encode(self, lead: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``(memory, pad_mask)`` for the lead input."""
        pad_mask = lead == self.pad_idx  # (B, L_lead)
        x = self._embed(lead)
        memory = self.encoder(x, src_key_padding_mask=pad_mask)
        return memory, pad_mask

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Input:  the dict that ``src.data.loaders.collate_satb`` emits.
        Output: {"s": Tensor, "a": Tensor, "t": Tensor, "b": Tensor}
                where each value has shape (B, L_voice, vocab_size).
        """
        memory, memory_pad_mask = self.encode(batch["lead"])

        logits: dict[str, Tensor] = {}
        for voice in _VOICE_KEYS:
            target = batch[voice]  # (B, L_voice)
            target_emb = self._embed(target)

            l_tgt = target.size(1)
            causal = nn.Transformer.generate_square_subsequent_mask(l_tgt).to(
                device=target.device
            )
            tgt_pad_mask = target == self.pad_idx

            decoded = self.decoder(
                tgt=target_emb,
                memory=memory,
                tgt_mask=causal,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=memory_pad_mask,
            )
            logits[voice] = self.out_proj[voice](decoded)
        return logits

    def num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

