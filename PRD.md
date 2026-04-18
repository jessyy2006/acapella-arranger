# Aca-Adapt — PRD v3

**Project:** Aca-Adapt
**Course:** CS 372 Introduction to Applied Machine Learning, Spring 2026 (Duke University)
**Deadline:** 2026-04-26, 11:59 pm
**Build window:** 5 days (2026-04-19 → 2026-04-23); buffer through submission
**Target score:** 103 / 100

---

## 1. Objective

A zero-cost, multi-stage ML pipeline that converts any input song into an editable four-part SATB (Soprano, Alto, Tenor, Bass) acapella arrangement, exported as MIDI / MusicXML and delivered through a deployed Streamlit web application.

## 2. Executive Summary

Aca-Adapt addresses the "black box" problem of contemporary music-generation AI: models produce uneditable audio that vocal groups cannot adapt to their own ensembles, keys, or skill levels. Aca-Adapt instead *deconstructs* source audio and *reconstructs* it into symbolic MIDI scores. Any input song becomes a rehearsal-ready, key-transposable, score-editable four-part arrangement usable in MuseScore, Finale, Sibelius, or any notation software.

## 3. Pipeline Architecture

### Stage 1 — Source Separation (audio → vocal stem)
- **Model:** HT-Demucs (Meta, pretrained, frozen) — hybrid transformer-demucs
- **I/O:** `.mp3` / `.mp4` / `.wav` → isolated vocal waveform
- **Engagement:** Used (frozen pretrained)

### Stage 2 — Pitch Transcription (vocal → monophonic MIDI lead)
- **Model:** CREPE (pretrained, frozen) — deep convolutional pitch estimator
- **I/O:** vocal waveform → (pitch, confidence) @ 10 ms frames → quantized monophonic MIDI lead
- **Preprocessing:** spectrogram-based feature pipeline; confidence thresholding to remove breaths / background
- **Engagement:** Used (frozen pretrained)

### Stage 3 — SATB Harmony Arranger (lead → 4-part SATB) — **custom model**
- **Architecture:** Hybrid Transformer + LSTM (PyTorch)
  - Embedding: 256-dim on (pitch, duration) tokens
  - Transformer encoder: 4 layers, 4 heads, d_model = 256 (attention paradigm)
  - LSTM decoder: 2 layers, hidden = 256, teacher forcing (recurrent paradigm)
  - 4 parallel output heads → S / A / T / B token streams
- **Two-stage training:**
  - **Phase A — Pretrain on JSB Chorales** (~389 Bach SATB chorales via `music21.corpus.chorales`)
  - **Phase B — Fine-tune on jaCappella** (~32 Japanese a cappella songs)
- **Data augmentation:** ×12 key transposition + sliding-window chunking (8-bar segments) → ~8,000 effective training sequences
- **Loss:** token-level cross-entropy, summed over all four voice heads
- **Optimizer:** AdamW, `ReduceLROnPlateau`, dropout 0.1, early stopping on validation loss
- **Engagement:** **Developed** (custom hybrid architecture, custom training loop, custom tokenization)

### Stage 4 — Voice-Leading Post-Processor (rule-based)
- Enforces human vocal ranges (Bass ≥ E2, Tenor C3–A4, Alto G3–E5, Soprano C4–C6)
- Filters impossible intervals (> octave leaps, parallel fifths / octaves)
- Toggleable — used as an ablation control

### Stage 5 — Export & Deployment
- **music21** converts predicted token streams → `.mid` and `.musicxml` for download
- **Streamlit** web UI: file upload → progress bar → preview → download
- Deployed on Streamlit Community Cloud (free tier)
- **Production hardening:** `st.cache_data` caching, rate limiting, structured logging, health check endpoint, error-handling pages

## 4. Data

| Source | Role | Format | License / Access |
|---|---|---|---|
| **jaCappella corpus** | Phase B fine-tuning (Japanese modern a cappella) | MIDI | Academic, via HuggingFace |
| **JSB Chorales** | Phase A pretraining (Bach SATB chorales) | MIDI | Public domain, via `music21.corpus.chorales.Iterator()` |
| **YouTube audio via `yt-dlp`** | Inference-time real-world testing only | Audio | Test-time only — **not claimed as training data** |

Effective training pool after augmentation: ~420 source songs × 12 transpositions × sliding-window chunking ≈ **~8,000 training sequences.**

## 5. Evaluation

### Quantitative metrics (≥ 3, rubric #86)
1. **Note accuracy** — token-level exact match vs jaCappella ground truth (per voice + aggregate)
2. **Voice-leading violation rate** — % notes outside range + parallel-fifth / parallel-octave rate
3. **Harmonic consonance score** — ratio of consonant-to-dissonant intervals across simultaneous S/A/T/B frames

### Qualitative
- Render 3 worked examples in MuseScore; present in walkthrough video
- Side-by-side comparison: model output vs human-arranged jaCappella ground truth

### Error analysis (rubric #87)
- Identify worst 5 predictions by note-accuracy
- Visualize failure modes (mode collapse on long sustains, out-of-range bass predictions, etc.)
- Discuss in technical walkthrough

### Ablation (rubric #91, ≥ 2 axes)
| Axis | Variants | Purpose |
|---|---|---|
| Voice-leading post-processor | ON / OFF | Isolates post-processing contribution |
| Architecture | Hybrid (Transformer + LSTM) / pure Transformer | Isolates recurrent-decoder contribution |

The architecture axis also satisfies **rubric #88** (compared architectures quantitatively).

## 6. Scoring Plan — targeting 103 / 100

### Category 1 — Machine Learning (cap 73 pts)

**Foundational (5 items, 17 pts)**
- #0 Modular code with reusable functions/classes (3)
- #1 Train/val/test split with documented ratios (3)
- #2 Tracked and visualized training curves (3)
- #5 Regularization ≥ 2 techniques: dropout + early stopping (5)
- #86 ≥ 3 distinct evaluation metrics (3)

**Mid-tier (7 items, 49 pts)**
- #23 Adapted pretrained model across substantially different domains (JSB Bach chorales → Japanese a cappella) (7)
- #66 Cross-modal generation (audio → symbolic MIDI) (7)
- #67 Audio data preprocessing with spectrograms / MFCCs (7)
- #75 Multi-stage ML pipeline (Demucs → CREPE → harmony transformer) (7)
- #87 Error analysis with visualization of failure cases (7)
- #88 Compared model architectures quantitatively (7)
- #91 Ablation study varying ≥ 2 independent design choices (7)

**High-tier (3 items, 30 pts)**
- #81 Custom architecture combining multiple paradigms (Transformer + LSTM hybrid) (10)
- #82 Deployed model as functional web application with UI (10)
- #83 Production-grade deployment (caching, rate limiting, logging, monitoring, error handling) (10)

**Raw ML total: 96 pts → capped at 73.** Cushion: 23 pts.

### Category 2 — Following Directions (cap 15 pts)
Self-assessment (3) + 8 documentation items (8) + 2 videos (4) + ≥ 2 workshop days (2) = **17 → capped at 15.**

### Category 3 — Project Cohesion & Motivation (cap 15 pts)
All 9 cohesion items claimed — single unified goal, real-world problem, synergistic components, clean codebase, metrics aligned with objectives. **21 → capped at 15.**

### **Total: 73 + 15 + 15 = 103 / 100 ✓**

## 7. Risk-Adjusted Cushion

| Scenario | Drop | Raw loss | Cap impact |
|---|---|---|---|
| #83 production hardening slips | Drop #83 | −10 | 0 (86 raw > 73) |
| Pure-Transformer baseline doesn't finish | Drop #88 | −7 | 0 (89 raw > 73) |
| Domain-transfer story unclear | Drop #23 | −7 | 0 (89 raw > 73) |
| Hybrid fails to converge — fall back to pure Transformer | Swap #81 → #18 | −5 | 0 (91 raw > 73) |

**Up to two items can slip with zero final-score impact.**

## 8. 5-Day Schedule

| Day | Date | ML Owner | UI / Deploy Owner |
|---|---|---|---|
| **D1** | Sun 4/19 | Repo setup, Colab env, Demucs + CREPE end-to-end notebook on 1 song | Streamlit skeleton, layout wireframe |
| **D2** | Mon 4/20 | Load JSB + jaCappella via `music21`, unified tokenizer, augmentation (×12 transpose + sliding windows), DataLoader, train/val/test split | Backend integration stub: `run_pipeline(audio_path) → midi` |
| **D3** | Tue 4/21 | Build hybrid Transformer + LSTM; kick off Phase A pretraining on JSB | Hook Demucs + CREPE into Streamlit; file upload working |
| **D4** | Wed 4/22 | Phase B fine-tuning on jaCappella; train pure-Transformer baseline (for #88); voice-leading post-processor; evaluation metrics pipeline | Production polish: caching, rate limiting, logging, error pages |
| **D5** | Thu 4/23 | Ablations, error analysis, finalize eval report, write README / SETUP / ATTRIBUTION with evidence links | Deploy to Streamlit Cloud, end-to-end smoke tests |
| **Buffer** | Fri 4/24 – Sun 4/26 | Record demo + walkthrough videos, submit self-assessment, attend workshop days | |

## 9. Out of Scope (deliberately cut, no score impact)

- Lakh MIDI pretraining — wrong domain (pop/rock instrumental, not SATB vocal)
- Distributed training (#105)
- RLHF / preference alignment (#104)
- Reproducing or beating a published paper (#99 / #101)
- Training-scale YouTube scraping (#13 — yt-dlp used for inference testing only)

## 10. Tooling (all $0)

- **Language:** Python 3.10+
- **Libraries:** `torch`, `librosa`, `music21`, `streamlit`, `demucs`, `crepe`, `yt-dlp`
- **Hardware:** Google Colab free T4 GPU for training
- **Hosting:** Streamlit Community Cloud free tier
- **Version control:** GitHub
