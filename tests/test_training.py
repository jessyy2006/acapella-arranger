"""Tests for the Day-3 Chunk-2 training loop.

Scope
-----
- Config loading + deep-merge for both phase overrides.
- Per-voice CE loss masks PAD and applies the teacher-forcing shift.
- Checkpoint save/load round-trip (state dicts match) + resume priority
  (``last.pt`` beats ``--init-from``).
- One-step training actually reduces loss on a tiny synthetic batch
  (smoke).
- Seed determinism across two fresh models.
- Per-source ``prepare_data.py`` emits the right files.

The source-split test is skipped on dev machines that haven't downloaded
the jaCappella corpus (we still validate JSB-only output there).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.data.vocab import PAD, VOCAB_SIZE
from src.models.hybrid import SATBHybrid
from src.training.checkpoint import (
    BEST_FILENAME,
    LAST_FILENAME,
    build_state,
    load,
    resume_or_init,
    save,
)
from src.training.config import TrainConfig, load_config
from src.training.metrics import compute_loss
from src.training.train import build_model, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model_cfg_kwargs() -> dict[str, int | float | str]:
    return dict(
        cls="hybrid",
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_lstm_layers=1,
        n_decoder_layers=1,
        d_ff=64,
        dropout=0.0,
    )


def _tiny_model() -> SATBHybrid:
    return SATBHybrid(
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_lstm_layers=1,
        d_ff=64,
        dropout=0.0,
    )


def _synthetic_batch(
    batch_size: int = 2,
    lead_len: int = 12,
    voice_len: int = 14,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Random non-PAD tokens; final two positions of each voice are PAD."""
    device = device or torch.device("cpu")
    out: dict[str, torch.Tensor] = {}
    out["lead"] = torch.randint(5, VOCAB_SIZE, (batch_size, lead_len), device=device)
    out["lead_len"] = torch.tensor([lead_len] * batch_size, device=device)
    for voice in ("s", "a", "t", "b"):
        toks = torch.randint(5, VOCAB_SIZE, (batch_size, voice_len), device=device)
        toks[:, -2:] = PAD
        out[voice] = toks
        out[f"{voice}_len"] = torch.tensor([voice_len - 2] * batch_size, device=device)
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_merge_both_phases(self) -> None:
        cfg_path = PROJECT_ROOT / "configs" / "train.yaml"
        pre = load_config(cfg_path, phase="pretrain")
        fin = load_config(cfg_path, phase="finetune")

        # Overrides applied
        assert pre.checkpoint.run_name == "phase_a"
        assert fin.checkpoint.run_name == "phase_b"
        assert "train_jsb" in pre.data.processed_file
        assert "train_jacappella" in fin.data.processed_file
        # Fine-tune lr is 10x lower per spec
        assert fin.optim.lr < pre.optim.lr
        assert abs(fin.optim.lr * 10 - pre.optim.lr) < 1e-8

        # Unrelated keys preserved across merges
        assert pre.seed == fin.seed == 42
        assert pre.model.d_model == fin.model.d_model

    def test_rejects_unknown_phase(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "c.yaml"
        yaml_path.write_text("seed: 1\nphases:\n  pretrain: {}\n")
        with pytest.raises(KeyError):
            load_config(yaml_path, phase="not_a_phase")

    def test_rejects_unknown_key(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "c.yaml"
        yaml_path.write_text("nonsense_key: 7\n")
        with pytest.raises(TypeError):
            load_config(yaml_path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestLoss:
    def test_masks_pad_positions(self) -> None:
        """PAD positions must not contribute to loss.

        Build two batches identical except at positions that will be PAD
        after the shift. Losses must match (pad positions are masked out).
        """
        torch.manual_seed(0)
        batch_a = _synthetic_batch()
        batch_b = {k: v.clone() for k, v in batch_a.items()}
        # Change what will be target[:, 1:] at PAD positions (the last token
        # of each voice in batch_a is PAD; shifting makes that the ignored
        # position). Flip to a non-PAD token — loss should not change.
        for v in ("s", "a", "t", "b"):
            batch_b[v][:, -1] = 7  # was PAD, is now a valid non-PAD token
            # And restore the *original* to PAD so the mask still hides it:
            # - actually, we want to prove: positions that are PAD in GOLD
            #   do not contribute. So keep batch_a's final position as PAD
            #   and batch_b's as a different value. Loss must be identical
            #   IFF the mask works, since PAD is ignore_index.
        # Actually the simpler check: supply logits that differ only at the
        # PAD-aligned positions and verify loss equality.
        model = _tiny_model().eval()
        with torch.no_grad():
            logits_a = model(batch_a)
            logits_b = {k: v.clone() for k, v in logits_a.items()}
            # Perturb logits at positions where the *shifted target* is PAD.
            for v in ("s", "a", "t", "b"):
                logits_b[v][:, -2, :] += 100.0  # targets[:, 1:][:, -1] == PAD
        loss_a, _ = compute_loss(logits_a, batch_a, pad_idx=PAD)
        loss_b, _ = compute_loss(logits_b, batch_a, pad_idx=PAD)
        assert torch.allclose(loss_a, loss_b), (
            f"loss changed when perturbing PAD-target logits: {loss_a} vs {loss_b}"
        )

    def test_shift_semantics(self) -> None:
        """Logit at position ``t`` is scored against target at ``t+1``."""
        batch = _synthetic_batch(batch_size=1, voice_len=5)
        # Remove PAD for this test (so the mask doesn't hide anything).
        for v in ("s", "a", "t", "b"):
            batch[v][:] = torch.randint(5, VOCAB_SIZE, batch[v].shape)

        # Construct logits that put peak probability on target[:, 1:] at
        # every position (from logits[:, :-1, :]). With PAD ignored and no
        # PAD present, loss should be near-zero.
        logits: dict[str, torch.Tensor] = {}
        for v in ("s", "a", "t", "b"):
            target = batch[v]
            B, L = target.shape
            lg = torch.full((B, L, VOCAB_SIZE), -1e4)
            # logit at position t peaks on target[t+1]
            for t in range(L - 1):
                lg[0, t, target[0, t + 1]] = 1e4
            logits[v] = lg
        loss, per_voice = compute_loss(logits, batch, pad_idx=PAD)
        assert loss.item() < 1e-3, f"shift should produce near-zero loss, got {loss}"


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def _make_model_opt_sched(self) -> tuple[SATBHybrid, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1)
        return model, opt, sched

    def test_round_trip(self, tmp_path: Path) -> None:
        model, opt, sched = self._make_model_opt_sched()
        state = build_state(
            model=model, optimizer=opt, scheduler=sched, epoch=3, best_val_loss=1.23
        )
        path = tmp_path / "ckpt.pt"
        save(state, path)
        loaded = load(path)

        # Check scalars round-trip
        assert loaded["epoch"] == 3
        assert abs(loaded["best_val_loss"] - 1.23) < 1e-9
        # Check model state keys match
        assert set(loaded["model_state"].keys()) == set(model.state_dict().keys())
        # Check each tensor value matches
        for k, v in model.state_dict().items():
            assert torch.equal(loaded["model_state"][k], v), f"mismatch at {k}"

    def test_resume_beats_init_from(self, tmp_path: Path) -> None:
        """If both last.pt and --init-from exist, last.pt wins."""
        # Build two distinct weight sets.
        m_resume, opt_r, sch_r = self._make_model_opt_sched()
        m_init, _, _ = self._make_model_opt_sched()
        # Force the two models' weights to differ.
        with torch.no_grad():
            for p in m_init.parameters():
                p.add_(1.0)

        ckpt_dir = tmp_path / "run"
        ckpt_dir.mkdir()
        save(build_state(model=m_resume, optimizer=opt_r, scheduler=sch_r, epoch=5, best_val_loss=0.4),
             ckpt_dir / LAST_FILENAME)

        init_from_path = tmp_path / "init.pt"
        save(build_state(
            model=m_init,
            optimizer=torch.optim.AdamW(m_init.parameters(), lr=1e-3),
            scheduler=None,
            epoch=99, best_val_loss=99.0,
        ), init_from_path)

        # Fresh model + optimizer to receive the resume.
        m_target, opt_t, sch_t = self._make_model_opt_sched()
        start_epoch, best_val = resume_or_init(
            ckpt_dir=ckpt_dir,
            init_from=init_from_path,
            allow_resume=True,
            model=m_target,
            optimizer=opt_t,
            scheduler=sch_t,
        )
        assert start_epoch == 6  # epoch + 1
        assert abs(best_val - 0.4) < 1e-9
        # Weights should match m_resume, not m_init.
        for k, v in m_resume.state_dict().items():
            assert torch.equal(m_target.state_dict()[k], v), f"{k} did not come from last.pt"


# ---------------------------------------------------------------------------
# Training step sanity
# ---------------------------------------------------------------------------


class TestTrainingStep:
    def test_one_step_reduces_loss(self) -> None:
        """One optimizer step on a fixed batch should reduce loss."""
        seed_everything(0)
        model = _tiny_model()
        batch = _synthetic_batch()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

        model.train()
        logits = model(batch)
        loss_0, _ = compute_loss(logits, batch, pad_idx=PAD)
        loss_0_val = loss_0.item()

        opt.zero_grad()
        loss_0.backward()
        opt.step()

        # Re-evaluate on the same batch.
        with torch.no_grad():
            logits2 = model(batch)
            loss_1, _ = compute_loss(logits2, batch, pad_idx=PAD)
        assert loss_1.item() < loss_0_val - 1e-3, (
            f"one step didn't reduce loss: {loss_0_val} -> {loss_1.item()}"
        )

    def test_determinism(self) -> None:
        """Same seed -> same loss on two fresh models."""

        def one_step_loss() -> float:
            seed_everything(42)
            model = _tiny_model()
            batch = _synthetic_batch()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            logits = model(batch)
            loss, _ = compute_loss(logits, batch, pad_idx=PAD)
            opt.zero_grad()
            loss.backward()
            opt.step()
            return loss.item()

        assert one_step_loss() == one_step_loss()

    def test_build_model_dispatches_correctly(self) -> None:
        from src.training.config import ModelCfg
        hybrid = build_model(ModelCfg(cls="hybrid", d_model=32, n_heads=4, n_encoder_layers=1, n_decoder_lstm_layers=1, d_ff=64, dropout=0.0))
        baseline = build_model(ModelCfg(cls="baseline", d_model=32, n_heads=4, n_encoder_layers=1, n_decoder_layers=1, d_ff=64, dropout=0.0))
        assert hybrid.__class__.__name__ == "SATBHybrid"
        assert baseline.__class__.__name__ == "SATBBaseline"


# ---------------------------------------------------------------------------
# Per-source split sanity — runs the real prepare_data script on a tiny slice.
# ---------------------------------------------------------------------------


class TestPerSourceSplits:
    def test_prepare_data_emits_per_source_files(self, tmp_path: Path) -> None:
        """prepare_data --force --limit-jsb 2 should emit combined + _jsb
        files. The _jacappella files are emitted only if a jaCappella raw
        tree exists locally (on CI / fresh clone, it won't).
        """
        jac_root = PROJECT_ROOT / "data" / "raw" / "jacappella"
        jac_present = jac_root.exists()

        out_dir = tmp_path / "processed"
        cmd = [
            sys.executable, "scripts/prepare_data.py",
            "--out", str(out_dir),
            "--limit-jsb", "2",
            "--force",
        ]
        if not jac_present:
            # Use a non-existent jacappella root so the script skips it fast.
            cmd += ["--jacappella-root", str(tmp_path / "nonexistent")]

        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stderr or result.stdout

        # Combined + JSB-only files always emitted.
        for split in ("train", "val", "test"):
            assert (out_dir / f"{split}.pt").exists(), result.stdout
            assert (out_dir / f"{split}_jsb.pt").exists(), result.stdout

        if jac_present:
            # jaCappella-only files should be produced alongside the others.
            for split in ("train", "val", "test"):
                assert (out_dir / f"{split}_jacappella.pt").exists(), result.stdout

        # Loaded JSB-only dataset should be non-empty.
        from src.data.loaders import load_dataset
        jsb_train = load_dataset(out_dir / "train_jsb.pt")
        assert len(jsb_train) > 0
