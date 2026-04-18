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

- **JSB Chorales** — 348 clean 4-voice Bach chorales, used to pre-train the model on general SATB structure. Public domain, loaded via `music21.corpus`.
- **jaCappella corpus** — 50 Japanese a cappella pieces with separate lead/SATB parts. Used to fine-tune for modern acapella style. Gated on HuggingFace — you'll need to accept the license at https://huggingface.co/datasets/jaCappella/jaCappella.

## Stack

Python 3.10 · PyTorch · music21 · Demucs · torchcrepe · Streamlit · Google Colab or Duke DCC for training · Streamlit Community Cloud for hosting. All $0.

## Current status (2026-04-18)

- **Day 1 shipped**: repo scaffold, data exploration, download scripts. Commits on `main`-track branches.
- **Day 2 shipped**: full data pipeline (tokenizer, vocab, augmentation, SATBDataset, DataLoader, prepare_data script, sanity notebook). 89 tests passing. Verified end-to-end by exporting and listening to generated MIDI.
- **Day 3 in progress**: model architecture shipped as of this writing (`src/models/hybrid.py`) — 7.8M-param Transformer encoder + four LSTM decoder heads with cross-attention. Training loop + runs still to come.
- **Open for review**: PR #1 at https://github.com/jessyy2006/acapella-arranger/pull/1 — Day 1 + Day 2 combined. Your review is welcome but not blocking.

## Who does what (agreed 2026-04-18)

Rough load split: Jess ~65%, you ~35%. Jess is taking the UI and final integration because the acapella domain intuition (what makes a good exported arrangement? what knobs should the user get?) is his strongest contribution.

### Jess owns

| Lane | Files / artifacts |
|---|---|
| Day 1–2 data pipeline (shipped) | `src/data/*`, `scripts/prepare_data.py`, `tests/test_*.py` except `test_models.py` |
| Day 3 model architecture + training code | `src/models/hybrid.py`, `src/training/*`, `configs/train.yaml`, `notebooks/03_train.ipynb` |
| Day 4/5 final integration | `src/pipeline/run_pipeline.py` — wires Demucs + torchcrepe + trained model + voice-leading post-process + music21 export into one callable |
| Day 4/5 voice-leading post-processor | `src/postprocess/voice_leading.py` — enforces SATB ranges, flags parallel fifths/octaves, smooths impossible leaps |
| Day 5 Streamlit app + deploy | `src/app/main.py`, Streamlit Cloud deployment |
| Day 5 demo video (non-technical, UX-focused, ~2 min) | `videos/demo.mp4` |
| Day 5 README + self-assessment lead | `README.md` final polish, Gradescope self-assessment submission |

### You own

| Lane | Files / artifacts |
|---|---|
| Day 3 training wall-clock runs on Colab (~4h combined) | Kick off Phase A (JSB pretrain) and Phase B (jaCappella fine-tune) once Jess ships the training loop; report metrics + curves back |
| Day 3 baseline model for ablation comparison | `src/models/baseline.py` — a pure-Transformer seq2seq (no LSTM) on the same inputs. Standalone, ~200 lines, uses the same tokenizer and DataLoader Jess built. Rubric: +3 pts (baseline for comparison) |
| Day 4 audio pipeline (audio → MIDI lead) | `src/pipeline/audio_to_midi.py` — HT-Demucs call to isolate lead vocal, torchcrepe to pitch-track it, quantise to our tokenizer's 16th-note grid. Return a music21 Part or our token format |
| Day 4 evaluation + ablation study | `src/eval/evaluate.py`, `src/eval/ablation.py`, `reports/*.md`, `reports/*.png`. Metrics: per-voice token accuracy, bar-wise accuracy, range compliance, duration-bucket accuracy. Ablation sweeps: hybrid vs baseline, with/without voice-leading rules, JSB-only vs JSB+jaCappella |
| Day 5 technical walkthrough video (~5 min, ML + audio + eval) | `videos/technical_walkthrough.mp4` |
| Day 5 README first draft of evaluation + methodology sections | Jess polishes; you own the evidence-driven parts |

Both of us touch the `Individual Contributions` section of the README — it's a rubric-required accounting of who did what.

## Onboarding steps (Cursor or whatever editor you use)

1. **Clone + env**: follow `SETUP.md` Path A (Duke DCC) or Path B (Colab) or Path B' (local conda). `pip install -r requirements.txt` gets you everything including pytest.
2. **Run the test suite**: `pytest` should show ~97 passing in ~3 seconds. If not, your env is broken.
3. **Run `prepare_data.py`**: `python scripts/prepare_data.py` produces `data/processed/{train,val,test}.pt`. Verifies data access end-to-end.
4. **Skim the docs in this order**: `README.md` → `docs/PRD.md` → `docs/PARTNER_BRIEF.md` (this file) → `ATTRIBUTION.md` → `SETUP.md`. ~20 min total read.
5. **Read your lane's source** before writing anything:
   - For the audio pipeline: `src/data/tokenizer.py` + `src/data/vocab.py` (you'll be outputting tokens in this format) + `scripts/download_data.py` (for Demucs' expected audio format)
   - For eval + ablation: `src/data/loaders.py` (for `collate_satb` + `load_dataset`) + `src/models/hybrid.py` (understand what you're evaluating) + `tests/test_models.py` (shape examples)
   - For baseline: `src/models/hybrid.py` (copy the forward-pass contract; strip the LSTM; use one decoder head that predicts all 4 voices or four heads as-is) + `tests/test_models.py` (mirror those tests)
6. **Point Cursor at context**: see below.

## Pointing Cursor at the right context

When you start a Cursor session for a specific lane, attach these files as context (Cursor's `@` reference or paste into the chat) so it has the same picture we've been building:

**For any task** — always attach these first:
- `docs/PRD.md` (product spec)
- `docs/PARTNER_BRIEF.md` (this file)
- `ATTRIBUTION.md` (scope boundaries — what's AI-assisted vs owned)
- `README.md` (high-level framing)

**If you're working on the baseline model** (`src/models/baseline.py`):
- `src/models/hybrid.py` (contract to match)
- `tests/test_models.py` (tests to mirror)
- `src/data/loaders.py` (batch shape)
- `src/data/vocab.py` (token grammar)

**If you're working on the audio pipeline** (`src/pipeline/audio_to_midi.py`):
- `src/data/tokenizer.py` (output format)
- `src/data/vocab.py` (what tokens mean)
- `notebooks/02_tokenizer_sanity.ipynb` (example of the tokenizer round-trip)
- Relevant Demucs + torchcrepe docs (external)

**If you're working on eval + ablation** (`src/eval/`):
- `src/models/hybrid.py` (forward contract)
- `src/data/loaders.py` (`load_dataset`, `collate_satb`)
- `data/processed/val.pt`, `test.pt` (the actual data)
- Any training-loop code Jess has shipped at that point

A `.cursorrules` file at repo root may also help Cursor keep project context across sessions — Jess can add one if you want; ping him.

## Staying in sync

- **Branching**: one branch per lane, named `feat/<area>-<short-desc>` (e.g., `feat/baseline-model`, `feat/audio-pipeline`, `feat/evaluation`). Cut from `main`.
- **PR cadence**: open a draft PR as soon as you have *something* committed, even if it's not done. Lets Jess see shape early + catch design mismatches before they cost rework.
- **Commits**: natural-language messages, no `feat:` / `fix:` prefix, no AI co-author footer. Match existing commit style (`git log --oneline` to see).
- **Tests**: every public function gets a test. `pytest` must stay green on your branch before PR.
- **Merge conflicts**: if you and Jess are both touching `src/data/loaders.py` or `src/models/hybrid.py` — rare, but possible — ping each other first. Rebase onto `main` (not merge commits) before asking for review.

## 5-day timeline (remaining)

Today is **Sat 2026-04-18**. Due **Sun 2026-04-26 at 11:59 pm**. 8 days left.

| Date | Jess | You |
|---|---|---|
| **Sun 4/19** | Day 3 chunks 2 (training loop) | Clone + env + tests passing + read docs + start on `baseline.py` |
| **Mon 4/20** | Day 3 chunk 3 (Phase A pretrain kicked off) | Finish `baseline.py`, review its PR |
| **Tue 4/21** | Day 3 chunk 4 (Phase B fine-tune), start audio integration | Start `audio_to_midi.py` (Demucs + torchcrepe) |
| **Wed 4/22** | Final integration (`run_pipeline.py`) + voice-leading post-process | Finish audio pipeline, start `evaluate.py` |
| **Thu 4/23** | Streamlit app | Ablation runs + plots; first-draft README |
| **Fri 4/24** | Streamlit deploy, demo video | Technical walkthrough video, polish report docs |
| **Sat 4/25** | README polish + self-assessment | Final review, any last fixes |
| **Sun 4/26** | Submit by 11:59 pm | Submit by 11:59 pm |

## Grading

Out of 100 with up to 103 possible. We're going for 103 by hitting 15 ML rubric items that fall out of this project naturally (custom architecture, multi-stage pipeline, cross-modal audio→symbolic translation, ablation study, deployed web app, etc.) plus full credit on documentation and cohesion categories. Full scoring breakdown in `docs/PRD.md`.

## Getting started right now

If you have an hour and want to move:

1. Do the `pytest` + `prepare_data.py` smoke in step 2–3 above.
2. Open `notebooks/02_tokenizer_sanity.ipynb` and run it top-to-bottom. You'll listen to a generated MIDI sanity file; if it sounds like roughly-Bach, the pipeline is validated end-to-end.
3. Start on `src/models/baseline.py`. Mirror the shape contract in `src/models/hybrid.py` but use `nn.TransformerDecoder` instead of `nn.LSTM` for the decoder side. One-head-per-voice or shared head — your call; document which in the docstring.

Ping Jess with questions.
