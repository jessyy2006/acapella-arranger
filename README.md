---
title: Aca-Adapt
emoji: 🎼
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Aca-Adapt

Convert any song into an editable four-part SATB (Soprano, Alto, Tenor, Bass) a cappella arrangement. Upload an audio file and download a MIDI score that opens in any notation software for further editing.

---

## What it Does

Aca-Adapt is a multi-stage machine-learning pipeline that takes an audio file (MP3 / WAV / MP4) and produces an editable four-part vocal arrangement as a 5-track MIDI file (Lead + Soprano + Alto + Tenor + Bass). The system isolates the lead vocal with HT-Demucs, transcribes it to a monophonic lead line with torchcrepe, and then uses a custom Transformer-encoder + per-voice-LSTM-decoder hybrid model — pretrained on Bach chorales (JSB) and fine-tuned on the jaCappella corpus — to predict harmonizing soprano, alto, tenor, and bass lines. A rule-based voice-leading post-processor keeps notes in human vocal ranges and flags textbook-forbidden parallels. The arrangement is delivered through a deployed Streamlit web application where users upload audio and download a `.mid` file ready to open in MuseScore, GarageBand, Finale, Sibelius, or any notation software.

## Quick Start

```bash
# 1. Clone and enter
git clone <repo-url> && cd "Acapella Arranger"

# 2. Create env and install (Python 3.10 or 3.11)
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Download data (jaCappella + JSB chorales)
python scripts/download_data.py

# 4. Run the Streamlit app locally
streamlit run src/app/main.py
```

Full setup instructions (GPU, training, reproducing evaluation, troubleshooting): see [SETUP.md](SETUP.md).

## Video Links

- **Demo video (non-technical):** https://www.youtube.com/watch?v=uCR3n2ECQMs
- **Technical walkthrough:** https://www.youtube.com/watch?v=NTvXp2JXiAc

## Evaluation

Metrics reported on the jaCappella held-out test set (15% of songs, song-level split, seed=42). Numbers are reproduced from `reports/phase_*_jacappella_metrics.json` and `reports/parallels_count.json`.

### Domain adaptation: Phase A → Phase B

Fine-tuning lifts both raw token correctness and a structural musical constraint.

| Metric | Phase A (JSB pretrain only) | Phase B (after jaCappella fine-tune) |
|---|---:|---:|
| Mean token accuracy | 0.2327 | **0.3762** |
| Duration-token accuracy | 0.2809 | **0.4605** |
| Voice-crossing rate (lower better) | 0.0567 | **0.0134** |

### Constraint-focused evaluation: parallel motion

Generated 15 SATB samples per checkpoint and counted parallel-fifths / parallel-octaves between adjacent voices (textbook-forbidden in classical voice leading). The model was never trained with an explicit voice-leading loss, so a low count indicates the constraint was internalized from data.

| Checkpoint | Mean violations / sample | Total over 15 samples |
|---|---:|---:|
| `baseline_phase_a_jsb` (pure-Transformer baseline) | 1.33 | 20 |
| `phase_a_jsb` (hybrid, JSB only) | 0.40 | 6 |
| `phase_a_jacappella` (hybrid, JSB only, eval on jaCappella) | 0.47 | 7 |
| `phase_b_jsb` (hybrid, JSB → jaCappella, eval on JSB) | 0.20 | 3 |
| **`phase_b_jacappella`** (hybrid, fine-tuned, in-domain) | **0.07** | **1** |

### Where to find supporting evidence

- Loss curves: `reports/phase_a_loss.png`, `reports/phase_b_loss.png`
- Per-checkpoint metrics: `reports/*_metrics.{json,md}`
- Parallel-motion counts: `reports/parallels_count.{json,md}`
- Ablation comparison plot: `reports/plots/ablation.png`
- Reproduction commands: [SETUP.md §7 — Reproducing evaluation](SETUP.md)

## Individual Contributions

This is a partner project. Both partners contributed to project design, the PRD, ablation design, and the final write-up.

- **Jess Young** — overall technical lead. Wrote the data pipeline (`src/data/`), the hybrid Transformer + LSTM model and the pure-Transformer baseline (`src/models/`), the training loop (`src/training/train.py`), the audio-to-token pipeline (`src/pipeline/audio_to_midi.py`), the end-to-end run wrapper (`src/pipeline/run_pipeline.py`), the voice-leading post-processor (`src/postprocess/`), the evaluation harness and ablation runner (`src/eval/`), the production-hardened Hugging Face Space deployment (`Dockerfile`, `scripts/deploy_to_hf_space.sh`), the demo video, and the README + self-assessment.
- **Soham Jinsi** — ran the Phase A (JSB pretrain) and Phase B (jaCappella fine-tune) training jobs on Colab, built the Streamlit front end (`src/app/main.py`), led the data exploration phase (`notebooks/01_data_exploration.ipynb`), helped ideate the project, contributed to debugging across the codebase, collaboratively designed the Transformer + LSTM hybrid architecture, and recorded the technical walkthrough video.

---

## Additional documentation

- [SETUP.md](SETUP.md) — installation, training, reproduction
- [ATTRIBUTION.md](ATTRIBUTION.md) — pretrained models, datasets, libraries, AI-tool usage
- [docs/PRODUCTION_HARDENING.md](docs/PRODUCTION_HARDENING.md) — caching / rate limiting / logging / error handling, with file:line evidence
- [docs/specs/](docs/specs/) — module specs (audio pipeline, baseline model, evaluation, training)

## License

MIT for project code. Dataset and model licenses listed in [ATTRIBUTION.md](ATTRIBUTION.md).
