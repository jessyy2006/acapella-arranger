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

## Your lane contracts

The detailed contract for each of your deliverables lives in [`docs/specs/`](./specs/). **Read the spec for a lane before writing any code in that lane** — specs tell you the exact interface, required tests, acceptance criteria, and gotchas.

Start with [`docs/specs/README.md`](./specs/README.md) — it's the index and explains how to use the spec files. Then pick your next task:

| Your next task | Open this spec |
|---|---|
| Build the baseline model for ablation | [`docs/specs/baseline_model.md`](./specs/baseline_model.md) |
| Build the audio→tokens pipeline | [`docs/specs/audio_pipeline.md`](./specs/audio_pipeline.md) |
| Build the evaluation + ablation harness | [`docs/specs/evaluation.md`](./specs/evaluation.md) |
| Kick off training runs on Colab | [`docs/specs/training_runs.md`](./specs/training_runs.md) |

## Onboarding steps (Cursor or whatever editor you use)

1. **Clone + env**: follow `SETUP.md` Path A (Duke DCC) or Path B (Colab) or Path B' (local conda). `pip install -r requirements.txt` gets you everything including pytest.
2. **Run the test suite**: `pytest` should show ~97 passing in ~3 seconds. If not, your env is broken.
3. **Run `prepare_data.py`**: `python scripts/prepare_data.py` produces `data/processed/{train,val,test}.pt`. Verifies data access end-to-end.
4. **Skim the docs in this order**: `README.md` → `docs/PRD.md` → `docs/PARTNER_BRIEF.md` (this file) → `ATTRIBUTION.md` → `SETUP.md`. ~20 min total read.
5. **Open the spec for your current task** in `docs/specs/` and read it top-to-bottom. It tells you the interface contract, required tests, files to read, gotchas, and acceptance criteria. Don't write code before reading the spec.
6. **Point Cursor at context**: see below.

## Pointing Cursor at the right context

Cursor auto-loads **`.cursorrules`** at the repo root every session — it covers the persistent project conventions (stack, code style, commit format, test requirements). You shouldn't need to re-brief it each time you open the project.

At the **start of a Cursor session for a specific task**, attach these as context (Cursor's `@` reference or paste the path into the chat):

1. **Always** — `docs/PRD.md`, `docs/PARTNER_BRIEF.md` (this file), `docs/specs/<your-current-lane>.md`.
2. **The files your spec's "Files to read first" section lists** — each spec enumerates the source files you need. Attach those.
3. **The specific test file** you'll be modifying or mirroring (e.g., `tests/test_models.py` when building the baseline).

The spec for each lane already has a "Files to read first" section — treat it as your checklist. Don't over-attach: Cursor's context window still matters, and attaching the whole codebase dilutes what it pays attention to.

If Cursor's output ignores a convention in `.cursorrules` (e.g., adds an AI co-author footer to a commit message), remind it explicitly in the chat — auto-loaded rules sometimes get overridden by Cursor's defaults.

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
