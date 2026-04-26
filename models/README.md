# `models/` — Trained model artifacts

This directory is **gitignored** for trained weights — the SATB harmony checkpoint is too large for git and lives on Hugging Face Hub instead. Inference downloads it on first run.

## Runtime checkpoint loading

The deployed Streamlit app (`src/app/main.py`) resolves a checkpoint via this priority order:

1. **`ACA_ADAPT_CHECKPOINT`** env var pointing at an existing local file (dev override).
2. **`ACA_ADAPT_HF_REPO_ID`** env var (e.g. `<user>/aca-adapt-model`) — pulls `phase_b_final.pt` from that HF Hub repo and caches it under `~/.cache/aca-adapt/`. This is the Hugging Face Spaces code path.
3. **Fallback:** `checkpoints/phase_b/phase_b_final.pt` relative to the project root.

Loader implementation: `_ensure_checkpoint` in `src/app/main.py` (process-cached via `@st.cache_resource` so the download runs once per container).

## Model definitions

The two model architectures are pure Python — no separately-saved configs needed:

- `src/models/hybrid.py` — `SATBHybrid`: Transformer encoder + per-voice unidirectional LSTM decoder with cross-attention onto the encoder memory. The main custom artifact for this project.
- `src/models/baseline.py` — `SATBBaseline`: pure-Transformer alternative with matched topology, used for the architecture ablation.

Hyperparameters (embed dim, layer counts, heads, dropout, batch size, LR, schedule) live in `configs/train.yaml` so the same recipe drives both Phase A pretrain and Phase B fine-tune via `--phase pretrain` / `--phase finetune`.

## Training output

A full training run writes checkpoints to `checkpoints/<run_name>/` — `last.pt` (resumable mid-run), `best.pt` (lowest val loss), `<run_name>_final.pt` (copy of best for downstream eval). See `src/training/checkpoint.py` for the save/restore contract.

## Uploading a freshly trained checkpoint to HF Hub

```bash
python scripts/upload_checkpoint_to_hf.py \
  --checkpoint checkpoints/phase_b/phase_b_final.pt \
  --repo-id <your-user>/aca-adapt-model
```
