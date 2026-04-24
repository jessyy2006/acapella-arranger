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

## Production Hardening

The deployed Streamlit app is hardened for a public Hugging Face Space rather than treated as a one-off demo. Concrete measures, each with a file:line pointer:

- **Caching at three scopes.** Process-wide checkpoint caching via `@st.cache_resource` (`src/app/main.py:_ensure_checkpoint`) downloads the 90 MB model weights once per container; upload probing via `@st.cache_data` (`src/app/main.py:_probe_duration`) avoids re-decoding an audio file on every Streamlit rerun; per-session result caching via `st.session_state` (`src/app/main.py` — the `_result` key) returns instantly when Generate is clicked on an unchanged `(file, options)` pair.
- **Per-session rate limiting.** Sliding-window limiter in `src/app/main.py:_rate_limit_check` blocks a single visitor from saturating the 2-vCPU free tier. Defaults to 3 generations per 10 minutes; override with `ACA_ADAPT_RATE_LIMIT_MAX` / `ACA_ADAPT_RATE_LIMIT_WINDOW_SEC` env vars.
- **Structured logging.** `_configure_logging` in `src/app/main.py` installs a key=value handler on the root logger. Every event (`upload`, `generate_start`, `generate_ok`, `generate_fail`, `rate_limited`, `probe_error`) is tagged with a short session id and emitted as one `grep`-able line. Running tallies of uploads / successes / failures are included on each `generate_ok` so the container log contains a rolling view of service health.
- **Explicit error handling on every failure mode.** `src/app/main.py:main` catches `FileNotFoundError` (checkpoint missing), `RuntimeError` (pipeline-internal), and a catch-all `Exception` path — each surfaces a distinct UI message *and* logs a `generate_fail` event with a classified reason. Module-load `ImportError`s are caught separately and rendered as a structured diagnostic banner (`_render_env_error`) instead of a stack trace.
- **Monitoring via log-scrapeable events.** Because every event line is key=value, grader-friendly queries work with plain shell tools, e.g. `grep 'event=generate_ok' logs | awk -F'elapsed_sec=' '{print $2}' | awk '{print $1}'` yields a distribution of end-to-end generation latencies. The container log on HF is the monitoring backend — no external dashboard required for a demo-tier deploy.
- **Environment-driven config.** Model repo (`ACA_ADAPT_HF_REPO_ID`), checkpoint filename (`ACA_ADAPT_HF_FILENAME`), Demucs model (`ACA_ADAPT_DEMUCS_MODEL`), and rate-limit parameters are all env vars with documented defaults in `src/app/main.py`. Swapping environments needs no code change.
- **Reproducible container image.** The `Dockerfile` pins `torch==2.3.1 torchaudio==2.3.1` from the CPU wheel index with a post-install force-reinstall + version assertion so a broken resolver fails the build loudly rather than producing a mismatched runtime. Deployment is driven by `scripts/deploy_to_hf_space.sh`, which pushes an orphan snapshot that excludes binary rubric evidence HF would otherwise reject.

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
