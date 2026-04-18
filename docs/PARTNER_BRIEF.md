# Aca-Adapt — Partner Brief

Quick rundown so you can jump in. Full technical spec is in `docs/PRD.md`; this is the summary.

## What we're building

Upload any song, get back an editable four-part (soprano, alto, tenor, bass) a cappella arrangement as a MIDI / MusicXML score. The user downloads it and opens it in MuseScore, Finale, or Sibelius to tweak.

The point: most music-generation AI spits out locked audio. Vocal groups need something they can actually edit — transpose the key, swap voices, adapt for their specific ensemble. Our pipeline keeps everything symbolic so the output is editable.

## How it works

```
audio upload (mp3/wav/mp4)
    ↓
HT-Demucs            → isolate the lead vocal from the mix          (pretrained, frozen)
    ↓
CREPE (torchcrepe)   → vocal audio → monophonic MIDI lead line      (pretrained, frozen)
    ↓
Harmony model        → lead → predicted soprano/alto/tenor/bass     (we train this)
   (Transformer + LSTM hybrid, PyTorch)
    ↓
Voice-leading rules  → clean up impossible intervals, out-of-range notes
    ↓
music21 export       → .mid + .musicxml
    ↓
Streamlit web app    → user uploads, downloads the result
```

Stages 1, 2, 4, 5 are off-the-shelf. Stage 3 is the model we build and train — that's the core contribution.

## Data

- **JSB Chorales** — ~389 Bach 4-voice chorales, used to pre-train the model on general SATB structure. Public domain, loaded via `music21.corpus`.
- **jaCappella corpus** — 35 Japanese a cappella pieces with separate lead/SATB parts. Used to fine-tune for modern acapella style. Gated on HuggingFace — you'll need to accept the license at https://huggingface.co/datasets/jaCappella/jaCappella.

## Stack

Python 3.10 · PyTorch · music21 · Demucs · torchcrepe · Streamlit · Google Colab or Duke DCC for training · Streamlit Community Cloud for hosting. All $0.

## Who does what (proposal, flex as needed)

**Me:**
- Data loading, tokenization, augmentation (JSB + jaCappella)
- Harmony model architecture + training loop
- Evaluation metrics, ablation study, error analysis

**You:**
- Streamlit app (file upload, progress bar, preview, download)
- Integration layer: wire Demucs + CREPE + our trained model into a single `run_pipeline(audio)` function the app calls
- Production polish: caching, rate limiting, structured logging, error handling
- Deploy to Streamlit Community Cloud

We'll both do: README, demo video, technical walkthrough video, final writeup. Individual Contributions section in the README reflects what each of us actually shipped.

## 5-day timeline

| Day | What |
|---|---|
| Sun 4/19 | Data download + exploration (data ready in `data/raw/`, stats in notebook) |
| Mon 4/20 | Tokenizer + DataLoader |
| Tue 4/21 | Hybrid model built; JSB pre-training kicks off. You: Streamlit skeleton |
| Wed 4/22 | Fine-tune on jaCappella; train baseline; voice-leading rules. You: production hardening |
| Thu 4/23 | Ablations, error analysis, README. You: deploy + smoke test |
| Fri–Sun | Buffer. Record videos, submit self-assessment |

Due **Sunday 2026-04-26, 11:59 pm**.

## Grading

Out of 100 with up to 103 possible. We're going for 103 by hitting 15 ML rubric items that fall out of this project naturally (custom architecture, multi-stage pipeline, cross-modal audio→symbolic translation, ablation study, deployed web app, production-grade deployment, etc.) plus full credit on documentation and cohesion categories. Full scoring breakdown in `docs/PRD.md`.

## Getting started

Setup instructions in `SETUP.md`. If you want to start right now, the data exploration notebook is `notebooks/01_data_exploration.ipynb`. Clone the repo, follow SETUP Path A (Duke DCC) or Path B (Colab) depending on where you want to work, and open that notebook.

Ping me with questions.
