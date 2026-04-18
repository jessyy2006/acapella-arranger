# Spec — Training Runs (Colab Operator Runbook)

## Goal

Execute two training runs on Google Colab (free T4 GPU), save the resulting model checkpoints, capture loss curves, and report back. This is **not** a coding task — Jess writes the training loop. Your job is the operator work: configure Colab, kick off the runs, handle disconnects, and return results.

## Rubric justification

Running training doesn't directly earn rubric points, but it **enables** the ~25 points of ML-category claims that depend on a trained model:

- Custom architecture (10 pts) — needs to actually be trained
- Training curves (3 pts) — needs a curve to show
- LR scheduling (3 pts) — needs to be exercised
- Checkpoint saving (implicit requirement for deployed model)
- Phase-A-then-Phase-B regime (part of the ablation study)

Without your two runs, most of Jess's code is dead weight.

## Prerequisites (Jess ships these first)

- `src/training/train.py` — the CLI training loop
- `configs/train.yaml` — hyperparameters (the two phases will use two different config values)
- `notebooks/03_train.ipynb` — Colab-friendly wrapper that calls `train.py` with the right flags

Once those are on `main` (or on a branch you can pull), you can start. Coordinate timing with Jess on Slack / DM.

## Phase A — JSB pretrain

### What you're running

Training the hybrid on JSB Chorales only (278 train / 59 val songs after split). Goal: teach the model general SATB structure before showing it jaCappella.

### Steps

1. **Open the training notebook** — `notebooks/03_train.ipynb` in Colab.
2. **Enable GPU** — Runtime → Change runtime type → T4 GPU.
3. **Clone the repo** in the first cell (Jess's notebook should have this template):
   ```python
   !git clone https://github.com/jessyy2006/acapella-arranger.git
   %cd acapella-arranger
   !git checkout feat/day-3-model  # or whichever branch has the training loop
   !pip install -q -r requirements.txt
   ```
4. **Authenticate HuggingFace** for the jaCappella download (needed even in Phase A, because `prepare_data.py` loads both corpora):
   ```python
   from huggingface_hub import notebook_login
   notebook_login()
   ```
   Paste your HF read token from https://huggingface.co/settings/tokens.
5. **Download data + prepare the split**:
   ```python
   !python scripts/download_data.py
   !python scripts/prepare_data.py
   ```
6. **Launch Phase A training** — should be a one-liner via the notebook. Something like:
   ```python
   !python -m src.training.train --config configs/train.yaml --phase pretrain
   ```
7. **Mount Google Drive** before step 6 so checkpoints persist if Colab disconnects:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !mkdir -p /content/drive/MyDrive/aca-adapt/checkpoints
   ```
   Point the training config's checkpoint dir at this mounted path (Jess will document this in the config).
8. **Watch for disconnects**. Colab disconnects after ~90 min idle or ~12 hr continuous. Keep the browser tab visible. If it disconnects, re-run from step 3 (the repo clone is idempotent) and resume from the last checkpoint.

### Expected wall-clock

~2 hours on T4 at batch_size=16, ~25 epochs over the JSB split. If it takes dramatically longer, check you're on T4 (not K80) and `torch.cuda.is_available()` returns `True`.

### What to save

- `phase_a_final.pt` (state dict) → Google Drive + share link with Jess
- `phase_a_loss.csv` (per-epoch train+val loss) → same
- `phase_a_loss.png` (training curve plot) → same
- Short note: final val loss, best val loss, wall-clock time, any anomalies

## Phase B — jaCappella fine-tune

### What you're running

Loading the Phase A checkpoint and continuing training on jaCappella only (the 50-song corpus, split 70/15/15 by song). Goal: adapt the JSB-pretrained model to modern acapella style while preserving the SATB structure it already learned.

### Steps

Same shape as Phase A but with different flags:

```python
!python -m src.training.train \
  --config configs/train.yaml \
  --phase finetune \
  --init-from /content/drive/MyDrive/aca-adapt/checkpoints/phase_a_final.pt
```

Lower learning rate (Jess's config will handle this — typically 10× lower than pretraining). Fewer epochs (jaCappella is small; overfits fast).

### Expected wall-clock

~1 hour on T4. Faster than Phase A because the dataset is smaller.

### What to save

Same artifacts as Phase A, prefixed `phase_b_` and stored in the same Drive folder.

## Baseline + ablation runs

Once Jess and you have both shipped (Jess: hybrid training; you: `src/models/baseline.py`), run these additional configurations:

1. **Baseline, same two-phase regime** — swap `SATBHybrid` for your `SATBBaseline` in the training code. Jess's training loop should accept a `--model-class` flag; if not, a small PR to add it is fine.
2. **JSB-only run** (for the pretraining ablation axis) — train the hybrid on JSB only, no Phase B fine-tune. Used to show Phase B actually helps.
3. **jaCappella-only run** — train the hybrid from random init on jaCappella only, no Phase A pretrain. Used to show Phase A actually helps.

Total: 5 training runs (2 hybrid phases + 2 baseline phases + 2 ablation runs — though you can parallelise if you have multiple Colab notebooks open on different Google accounts).

### What to save (per ablation run)

Same three artifacts per run, named per the config (e.g. `baseline_phase_a.pt`, `hybrid_jsb_only.pt`).

## Gotchas

- **T4 OOM at large batch sizes.** If you see CUDA OOM, halve batch_size in the config. Default should be 16; dropping to 8 is safe.
- **Colab disconnects on idle.** Use the `pretraining_keepalive` trick: open the browser console (F12) and run a JS snippet that clicks the "reconnect" button every 10 min. Or just keep the tab visible.
- **Drive rate limits.** If you `torch.save` a checkpoint every epoch, Drive may throttle. Save every 5 epochs and keep the best-val-loss one.
- **HuggingFace gated dataset.** You must have accepted the jaCappella license on https://huggingface.co/datasets/jaCappella/jaCappella **before** running `download_data.py`. First run will fail with a 403 otherwise.
- **Don't commit checkpoints to git.** `.gitignore` already excludes `models/*.pt` and `data/processed/*.pt`. Share checkpoints via Drive; Jess will download them into `models/` locally when she integrates.
- **Seed everything.** `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` at the top of the notebook. Otherwise ablation runs aren't comparable.
- **Validation loss plateauing is OK.** Deep nets on small datasets often plateau before overfitting shows up in val loss. Stop when train loss stops decreasing OR at the epoch budget, whichever comes first. Let Jess know which happened.

## Files to read first

1. **`docs/PRD.md`** — the training strategy section. Two-phase rationale, expected metric targets.
2. **`SETUP.md`** — specifically the "Running in a remote Jupyter environment → Path B — Google Colab" section.
3. **`notebooks/03_train.ipynb`** — once Jess has shipped it. Read it end-to-end before running.
4. **`configs/train.yaml`** — once it exists. Understand what each hyperparameter does so you can diagnose issues.

## Acceptance criteria

- [ ] Phase A checkpoint + loss curves in Drive, shared with Jess.
- [ ] Phase B checkpoint + loss curves in Drive, shared with Jess.
- [ ] Short written report for each run (1 paragraph): final loss, wall-clock, any issues.
- [ ] Ablation runs completed (5 total) with matching artifacts.
- [ ] Loss curve PNGs copied into the repo at `reports/plots/` via a PR (these are small files, ~10 KB each, safe to commit).

## Out of scope

- **Writing the training loop.** That's `src/training/train.py` — Jess owns it.
- **Debugging the training loop.** If training crashes for a reason that looks like a code bug (not OOM, not config mistake), ping Jess with the full traceback. Don't try to fix her code.
- **Interpreting results.** That's the evaluation spec — once you have the checkpoints, pass them to yourself for `src/eval/evaluate.py` and write prose there.
