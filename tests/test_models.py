"""Forward-pass shape + invariance tests for SATBHybrid.

Scope is deliberately narrow for Chunk 1 — we're validating the model as a
pure function ``batch -> logits``. Losses, optimiser, and training dynamics
are exercised in Day 3 Chunk 2.
"""

from __future__ import annotations

import pytest
import torch
from music21 import corpus, stream

from src.data.dataset import SATBDataset
from src.data.load import is_clean_satb
from src.data.loaders import collate_satb
from src.data.vocab import PAD, VOCAB_SIZE
from src.models.hybrid import SATBHybrid


VOICES = ("s", "a", "t", "b")


def _make_synthetic_batch(
    batch_size: int = 2,
    lead_len: int = 20,
    voice_lens: dict[str, int] | None = None,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Hand-build a batch of random non-PAD tokens with padded tails."""
    voice_lens = voice_lens or {"s": 18, "a": 22, "t": 19, "b": 24}
    device = device or torch.device("cpu")

    def _padded(length: int, real_lengths: list[int]) -> torch.Tensor:
        # Random tokens in the valid pitch/duration range (skip PAD=0).
        x = torch.randint(1, VOCAB_SIZE, (batch_size, length), device=device)
        for i, real_len in enumerate(real_lengths):
            if real_len < length:
                x[i, real_len:] = PAD
        return x

    lead_real = [lead_len, lead_len - 3]
    out: dict[str, torch.Tensor] = {
        "lead": _padded(lead_len, lead_real),
        "lead_len": torch.tensor(lead_real, device=device),
    }
    for voice in VOICES:
        length = voice_lens[voice]
        real = [length, length - 4]
        out[voice] = _padded(length, real)
        out[f"{voice}_len"] = torch.tensor(real, device=device)
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model() -> SATBHybrid:
    # Small config keeps every test fast (<50 ms per forward pass).
    torch.manual_seed(0)
    return SATBHybrid(
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_lstm_layers=1,
        d_ff=128,
        dropout=0.0,  # deterministic forward for invariance tests
    ).eval()


# ---------------------------------------------------------------------------
# Construction + parameter accounting
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_config_is_reasonably_sized(self):
        # Guards against a configuration change that balloons the model.
        model = SATBHybrid()
        n_params = model.num_parameters()
        assert 3_000_000 < n_params < 15_000_000, (
            f"Expected 3-15M params, got {n_params:,}"
        )

    def test_rejects_mismatched_head_and_dim(self):
        with pytest.raises(ValueError):
            SATBHybrid(d_model=256, n_heads=5)


# ---------------------------------------------------------------------------
# Forward-pass shape
# ---------------------------------------------------------------------------


class TestForwardShape:
    def test_returns_logits_for_all_four_voices(self, tiny_model: SATBHybrid):
        batch = _make_synthetic_batch()
        logits = tiny_model(batch)
        assert set(logits.keys()) == set(VOICES)

    def test_logits_match_target_length_per_voice(self, tiny_model: SATBHybrid):
        voice_lens = {"s": 18, "a": 22, "t": 19, "b": 24}
        batch = _make_synthetic_batch(voice_lens=voice_lens)
        logits = tiny_model(batch)
        for voice, length in voice_lens.items():
            assert logits[voice].shape == (2, length, VOCAB_SIZE), (
                f"voice {voice}: expected (2, {length}, {VOCAB_SIZE}), "
                f"got {tuple(logits[voice].shape)}"
            )

    def test_logits_are_finite(self, tiny_model: SATBHybrid):
        batch = _make_synthetic_batch()
        logits = tiny_model(batch)
        for voice, lg in logits.items():
            assert torch.isfinite(lg).all(), f"voice {voice} produced non-finite logits"


# ---------------------------------------------------------------------------
# Padding invariance — encoder output at valid positions shouldn't change
# when we append extra PAD tokens at the tail of the lead.
# ---------------------------------------------------------------------------


class TestPaddingInvariance:
    def test_extra_pad_on_lead_leaves_valid_encoder_outputs_unchanged(
        self, tiny_model: SATBHybrid
    ):
        torch.manual_seed(7)
        lead_short = torch.randint(5, 133, (1, 12))  # 12 real tokens, no PAD
        lead_long = torch.cat(
            [lead_short, torch.full((1, 8), PAD, dtype=torch.long)], dim=1
        )  # same 12 real + 8 PAD

        mem_short, _ = tiny_model.encode(lead_short)
        mem_long, _ = tiny_model.encode(lead_long)

        # Outputs at real positions should match (up to float noise).
        diff = (mem_short - mem_long[:, :12, :]).abs().max().item()
        assert diff < 1e-4, f"max diff at valid positions = {diff}"


# ---------------------------------------------------------------------------
# Loss smoke — confirm model output plugs into cross-entropy
# ---------------------------------------------------------------------------


class TestLossSmoke:
    def test_cross_entropy_is_finite_over_padding(self, tiny_model: SATBHybrid):
        batch = _make_synthetic_batch()
        logits = tiny_model(batch)

        total_loss = torch.tensor(0.0)
        for voice in VOICES:
            target = batch[voice]
            # Standard next-token CE (no shift here — that's the training
            # loop's job — we just verify the shapes flow through).
            loss = torch.nn.functional.cross_entropy(
                logits[voice].reshape(-1, VOCAB_SIZE),
                target.reshape(-1),
                ignore_index=PAD,
            )
            assert torch.isfinite(loss)
            total_loss = total_loss + loss
        assert torch.isfinite(total_loss)


# ---------------------------------------------------------------------------
# Integration with the real collate — end-to-end from a real dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_batch() -> dict[str, torch.Tensor]:
    # Pull one real chorale to validate that collate_satb output actually
    # plugs into the model without shape surprises.
    chorale = None
    for score in corpus.chorales.Iterator():
        if is_clean_satb(score):
            chorale = score
            break
    if chorale is None:
        pytest.skip("no clean SATB chorale available")

    ds = SATBDataset([("jsb", chorale)], augment=False)
    return collate_satb([ds[i] for i in range(min(3, len(ds)))])


class TestRealCollateIntegration:
    def test_model_consumes_real_collate_output(
        self, tiny_model: SATBHybrid, real_batch: dict[str, torch.Tensor]
    ):
        logits = tiny_model(real_batch)
        batch_size = real_batch["lead"].shape[0]
        for voice in VOICES:
            expected = (batch_size, real_batch[voice].shape[1], VOCAB_SIZE)
            assert logits[voice].shape == expected
