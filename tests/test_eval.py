"""Tests for the evaluation harness.

Covers the four cases the spec names under ``docs/specs/evaluation.md`` →
"Required tests":

1. metric functions produce the right answer on hand-crafted inputs,
2. PAD positions are excluded from accuracy,
3. ``evaluate_checkpoint`` runs end-to-end on a tiny model + tiny dataset,
4. ``run_ablation`` errors clearly when a checkpoint path is missing.

Tiny-model / tiny-dataset fixtures mirror the ones in ``test_models.py``
and ``test_dataset.py`` so the whole file stays well under a few seconds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from music21 import corpus, stream

from src.data.dataset import SATBDataset
from src.data.load import is_clean_satb
from src.data.vocab import BAR, DUR_OFFSET, PAD, SOS, VOCAB_SIZE, pitch_to_token
from src.eval.ablation import run_ablation
from src.eval.evaluate import evaluate_checkpoint
from src.eval.metrics import (
    bar_accuracy,
    duration_bucket_accuracy,
    next_token_accuracy,
    per_voice_accuracy,
    pitch_count_by_voice,
    pitch_range_compliance,
    voice_crossing_rate,
)
from src.models.hybrid import SATBHybrid


VOICES = ("s", "a", "t", "b")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def jsb_score() -> stream.Score:
    """One clean SATB chorale — enough to round-trip a mini dataset."""
    for score in corpus.chorales.Iterator():
        if is_clean_satb(score):
            return score
    pytest.skip("no clean JSB chorales available")


@pytest.fixture(scope="module")
def tiny_hparams() -> dict[str, int | float]:
    return {
        "d_model": 32,
        "n_heads": 4,
        "n_encoder_layers": 1,
        "n_decoder_lstm_layers": 1,
        "d_ff": 64,
        "dropout": 0.0,
    }


def _perfect_logits_from_targets(targets: torch.Tensor) -> torch.Tensor:
    """Build ``(B, L, V)`` logits whose argmax equals a 1-step right-shift
    of ``targets`` — so under the teacher-forcing shift convention every
    non-PAD prediction is correct."""
    b, length = targets.shape
    logits = torch.full((b, length, VOCAB_SIZE), -1e9, dtype=torch.float32)
    # logits[:, t] should predict targets[:, t+1].
    for i in range(b):
        for t in range(length - 1):
            tgt_next = int(targets[i, t + 1].item())
            logits[i, t, tgt_next] = 10.0
        # Last position has no next token; fill with targets[i, -1] just so
        # argmax is defined.
        logits[i, -1, int(targets[i, -1].item())] = 10.0
    return logits


# ---------------------------------------------------------------------------
# 1. Metric correctness on known inputs
# ---------------------------------------------------------------------------


class TestKnownInputs:
    def test_next_token_accuracy_perfect_is_one(self):
        targets = torch.tensor([[SOS, pitch_to_token(60), DUR_OFFSET, pitch_to_token(62)]])
        logits = _perfect_logits_from_targets(targets)
        assert next_token_accuracy(logits, targets) == 1.0

    def test_next_token_accuracy_single_mistake(self):
        # 3 predictable next-token positions (len-1=3). One wrong → 2/3.
        targets = torch.tensor([[SOS, pitch_to_token(60), DUR_OFFSET, pitch_to_token(62)]])
        logits = _perfect_logits_from_targets(targets)
        # Flip the first prediction — favour a different token instead.
        wrong_tok = pitch_to_token(61)
        logits[0, 0, wrong_tok] = 100.0
        got = next_token_accuracy(logits, targets)
        assert math.isclose(got, 2 / 3, rel_tol=1e-6)

    def test_per_voice_accuracy_covers_all_voices(self):
        targets = torch.tensor([[SOS, pitch_to_token(60), DUR_OFFSET]])
        logits = _perfect_logits_from_targets(targets)
        out = per_voice_accuracy({v: logits for v in VOICES}, {v: targets for v in VOICES})
        assert set(out.keys()) == set(VOICES)
        for v in VOICES:
            assert out[v] == 1.0

    def test_bar_accuracy_perfect_matches(self):
        # SOS | pitch pitch BAR pitch pitch — 2 bars if we shift properly.
        targets = torch.tensor(
            [
                [
                    SOS,
                    pitch_to_token(60),
                    DUR_OFFSET,
                    BAR,
                    pitch_to_token(62),
                    DUR_OFFSET + 1,
                ]
            ]
        )
        logits = _perfect_logits_from_targets(targets)
        got = bar_accuracy({v: logits for v in VOICES}, {v: targets for v in VOICES})
        # With a correct shift, every bar matches.
        assert got == 1.0

    def test_bar_accuracy_zero_when_all_wrong(self):
        targets = torch.tensor(
            [
                [
                    SOS,
                    pitch_to_token(60),
                    DUR_OFFSET,
                    BAR,
                    pitch_to_token(62),
                    DUR_OFFSET + 1,
                ]
            ]
        )
        # All-wrong: argmax points at the wrong pitch at every position.
        logits = torch.full((1, 6, VOCAB_SIZE), -1e9, dtype=torch.float32)
        wrong = pitch_to_token(100)
        logits[:, :, wrong] = 10.0
        got = bar_accuracy({v: logits for v in VOICES}, {v: targets for v in VOICES})
        assert got == 0.0

    def test_duration_bucket_accuracy_selects_only_duration_gold(self):
        # Target has pitches and durations interleaved. duration_bucket_accuracy
        # should score only the positions where gold is a duration token.
        targets = torch.tensor(
            [
                [
                    SOS,
                    pitch_to_token(60),
                    DUR_OFFSET,
                    pitch_to_token(62),
                    DUR_OFFSET + 1,
                ]
            ]
        )
        logits = _perfect_logits_from_targets(targets)
        assert duration_bucket_accuracy({v: logits for v in VOICES}, {v: targets for v in VOICES}) == 1.0

    def test_pitch_range_compliance_in_and_out(self):
        # Each voice sees one in-range + one out-of-range pitch. Result 0.5
        # per voice, count 2.
        in_range = {"s": 60, "a": 55, "t": 55, "b": 40}
        out_of_range = {"s": 20, "a": 10, "t": 10, "b": 5}
        samples = {
            v: [[pitch_to_token(in_range[v]), pitch_to_token(out_of_range[v])]] for v in VOICES
        }
        compliance = pitch_range_compliance(samples)
        counts = pitch_count_by_voice(samples)
        for v in VOICES:
            assert compliance[v] == 0.5
            assert counts[v] == 2

    def test_voice_crossing_rate_zero_when_satb_ordering_holds(self):
        # Single sample each voice: pitch then duration-1 so timeline is length 1.
        samples = {
            "s": [[pitch_to_token(72), DUR_OFFSET]],
            "a": [[pitch_to_token(67), DUR_OFFSET]],
            "t": [[pitch_to_token(60), DUR_OFFSET]],
            "b": [[pitch_to_token(48), DUR_OFFSET]],
        }
        assert voice_crossing_rate(samples) == 0.0

    def test_voice_crossing_rate_one_when_ordering_inverted(self):
        samples = {
            "s": [[pitch_to_token(48), DUR_OFFSET]],
            "a": [[pitch_to_token(60), DUR_OFFSET]],
            "t": [[pitch_to_token(67), DUR_OFFSET]],
            "b": [[pitch_to_token(72), DUR_OFFSET]],
        }
        assert voice_crossing_rate(samples) == 1.0

    def test_voice_crossing_rate_no_cross_sample_bleed(self):
        # Regression: the prior implementation flattened all samples into one
        # stream, so the last pitch of sample 0 was compared against the first
        # of sample 1. Here each sample individually satisfies SATB ordering
        # but the concatenation does NOT — a correct per-sample implementation
        # returns 0.0.
        samples = {
            "s": [[pitch_to_token(72), DUR_OFFSET], [pitch_to_token(50), DUR_OFFSET]],
            "a": [[pitch_to_token(67), DUR_OFFSET], [pitch_to_token(45), DUR_OFFSET]],
            "t": [[pitch_to_token(60), DUR_OFFSET], [pitch_to_token(40), DUR_OFFSET]],
            "b": [[pitch_to_token(48), DUR_OFFSET], [pitch_to_token(30), DUR_OFFSET]],
        }
        assert voice_crossing_rate(samples) == 0.0


# ---------------------------------------------------------------------------
# 2. PAD positions must not skew accuracy
# ---------------------------------------------------------------------------


class TestPadExclusion:
    def test_pad_positions_dont_drag_down_accuracy(self):
        # Gold: SOS, 60, 62, PAD, PAD. Predictions at non-PAD are correct;
        # "predictions" at PAD-shifted positions point at a wrong token.
        # Shifted convention: logits[:, :-1] predicts targets[:, 1:], so the
        # last two target positions (1: 62, PAD) evaluate logits[:, 2] and
        # logits[:, 3]. PAD itself is masked out.
        targets = torch.tensor([[SOS, pitch_to_token(60), pitch_to_token(62), PAD, PAD]])
        logits = _perfect_logits_from_targets(targets)
        # Corrupt the prediction at the PAD-gold positions.
        wrong = pitch_to_token(100)
        logits[0, 2, :] = -1e9
        logits[0, 2, wrong] = 10.0
        logits[0, 3, :] = -1e9
        logits[0, 3, wrong] = 10.0
        # Gold at position 1 (pitch 60) and position 2 (pitch 62) are the only
        # non-PAD next-token positions. position 1 predicted from logits[:, 0]
        # (still correct), position 2 from logits[:, 1] (still correct).
        assert next_token_accuracy(logits, targets) == 1.0


# ---------------------------------------------------------------------------
# 3. End-to-end evaluate_checkpoint on tiny model + tiny dataset
# ---------------------------------------------------------------------------


class TestEvaluateCheckpointE2E:
    def test_returns_expected_keys(
        self, jsb_score: stream.Score, tiny_hparams: dict[str, int | float], tmp_path: Path
    ):
        torch.manual_seed(0)
        model = SATBHybrid(**tiny_hparams)
        ckpt_path = tmp_path / "tiny.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Sibling config so _build_model skips the configs/train.yaml fallback
        # (which would have the full-size hparams and fail to load our tiny
        # state_dict).
        (tmp_path / "tiny.config.json").write_text(json.dumps(tiny_hparams), encoding="utf-8")

        # A two-example test split — enough to exercise batching + metrics.
        dataset = SATBDataset([("jsb", jsb_score)], augment=False)
        if len(dataset) < 1:
            pytest.skip("JSB chorale produced zero windows for the tiny dataset")
        split_path = tmp_path / "test.pt"
        torch.save(dataset, split_path)

        out = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            test_dataset_path=split_path,
            batch_size=2,
            device="cpu",
            model_class="hybrid",
        )
        # All voice-keyed metrics plus the aggregates.
        for v in VOICES:
            assert f"acc_{v}" in out
            assert f"range_{v}" in out
            assert f"pitch_count_{v}" in out
        for key in ("acc_mean", "duration_acc", "bar_acc", "voice_crossing_rate"):
            assert key in out
        # Sanity: accuracies live in [0, 1] or are NaN.
        for v in VOICES:
            value = out[f"acc_{v}"]
            assert (0.0 <= value <= 1.0) or math.isnan(value)


# ---------------------------------------------------------------------------
# 4. Ablation surfaces a clear error on missing checkpoints
# ---------------------------------------------------------------------------


class TestAblationMissingCheckpoint:
    def test_missing_path_raises_filenotfound(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.pt"
        # Split path can be anything valid — ablation errors before opening it.
        dummy_split = tmp_path / "split.pt"
        dummy_split.write_bytes(b"")
        with pytest.raises(FileNotFoundError) as excinfo:
            run_ablation({"missing_run": missing}, dummy_split)
        # Error message should name the missing run and the missing path so
        # the user can diagnose without a stack-trace.
        assert "missing_run" in str(excinfo.value)
        assert str(missing) in str(excinfo.value)
