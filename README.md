# Aca-Adapt

Convert any song into an editable four-part SATB (Soprano, Alto, Tenor, Bass) a cappella arrangement. Upload an audio file and download a MIDI or MusicXML score that opens in any notation software for further editing.

---

## What it Does

Aca-Adapt is a multi-stage machine-learning pipeline that takes an audio file (MP3 / WAV / MP4) and produces an editable four-part vocal arrangement as MIDI and MusicXML. The system isolates the lead vocal with HT-Demucs, transcribes it to a monophonic lead line with CREPE, and then uses a custom Transformer + LSTM hybrid model — pretrained on Bach chorales and fine-tuned on the jaCappella corpus — to predict harmonizing soprano, alto, tenor, and bass lines. A rule-based voice-leading post-processor ensures notes stay in human vocal ranges and filters impossible intervals. The arrangement is delivered through a deployed Streamlit web application where users upload audio and download a `.mid` or `.musicxml` file ready to open in MuseScore, Finale, Sibelius, or any notation software.

## Quick Start

```bash
# 1. Clone and enter
git clone <repo-url> && cd "Acapella Arranger"

# 2. Create env and install
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Download data
python scripts/download_data.py

# 4. Run the Streamlit app locally
streamlit run src/app/main.py
```

Full setup instructions (GPU, training, reproducing evaluation): see [SETUP.md](SETUP.md).

## Video Links

- **Demo video (non-technical):** _TBD — will be linked before 2026-04-26_
- **Technical walkthrough:** _TBD — will be linked before 2026-04-26_

## Evaluation

Metrics reported on the jaCappella held-out test set (15% of songs, song-level split).

| Metric | Hybrid (T + LSTM) | Pure Transformer | Voice-Leading OFF |
|---|---|---|---|
| Note accuracy (aggregate) | _TBD_ | _TBD_ | _TBD_ |
| Voice-leading violation rate | _TBD_ | _TBD_ | _TBD_ |
| Harmonic consonance score | _TBD_ | _TBD_ | _TBD_ |

Ablation axes, error analysis, and qualitative examples are written up in `reports/` and walked through on video. See `notebooks/04_evaluation.ipynb` for the full reproduction pipeline.

## Individual Contributions

- **[Partner 1 name]:** _TBD — e.g., custom hybrid model architecture, training pipeline, data augmentation + tokenization, evaluation metrics, error analysis_
- **[Partner 2 name]:** _TBD — e.g., Streamlit UI, production hardening (caching, rate limiting, logging), deployment, Demucs + CREPE integration, UX_

Both partners contributed equally to project design, PRD, ablation design, and the final write-up.

## Project Structure

```
.
├── src/
│   ├── data/         # jaCappella + JSB loading, tokenization, augmentation
│   ├── models/       # Hybrid Transformer+LSTM, baseline Transformer
│   ├── pipeline/     # Demucs + CREPE end-to-end inference
│   ├── training/     # Train loop, optimizer, scheduling
│   ├── postprocess/  # Voice-leading rules, music21 export
│   ├── eval/         # Metrics, ablation, error analysis
│   └── app/          # Streamlit web UI
├── notebooks/        # Exploration, training, ablation, evaluation
├── configs/          # Training hyperparameter YAMLs
├── scripts/          # Data download and utility scripts
├── reports/          # Error analysis, evaluation write-ups
├── data/             # (git-ignored) raw + processed data
├── models/           # (git-ignored) checkpoints
└── outputs/          # (git-ignored) generated MIDI + logs
```

## Documentation

- [docs/PRD.md](docs/PRD.md) — product requirements, architecture, scoring plan
- [SETUP.md](SETUP.md) — installation, training, reproduction
- [ATTRIBUTION.md](ATTRIBUTION.md) — pretrained models, datasets, libraries, AI-tool usage

## License

MIT for project code. Dataset and model licenses listed in [ATTRIBUTION.md](ATTRIBUTION.md).
