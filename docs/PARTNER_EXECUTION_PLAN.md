# Partner Execution Plan — Soham's Remaining Work

**Overall Progress:** `0%`

**Owner:** Soham (execute via Cursor)
**Hard deadline:** end of **Tuesday night** — everything in this plan completed and on `main` by then.
**Submission deadline:** Sun 2026-04-26, 11:59 pm (buffer days after this plan lands are for Jess's integration + final polish only).

## TLDR

Four lanes, **all due by end of Tuesday night**. Compressed from the original 5-day timeline because we need finished weights + eval numbers before Jess can start the deployed-app integration. Jess shipped the training loop (PR #7, awaiting your review/merge) and the model architecture. Your lanes:

1. **Training runs** (wall-clock Colab, parallelizable across Google accounts) — produces trained checkpoints.
2. **Audio pipeline** — `src/pipeline/audio_to_midi.py`: converts an uploaded song to the lead-melody tokens the model consumes.
3. **Evaluation + ablation** — `src/eval/evaluate.py` + `src/eval/ablation.py`: metrics, comparison plots across all 5 runs.
4. **First-draft README methodology + evaluation sections** — evidence-driven prose tied to the plots + numbers from lane 3.

**Cut from original scope (do AFTER the Tuesday deadline if time allows):** technical walkthrough video, per-run markdown notes, final review pass. The video can slip to Thu/Fri — the code + eval + checkpoints cannot.

The spec files in `docs/specs/` are **the contract** for each lane. Read the spec for a lane before writing any code in that lane. If a spec is silent on a detail, pick the default that matches existing code patterns (see `.cursorrules` + existing `src/` modules).

## Parallelization plan

Training is wall-clock (GPU time, not human time). Code while training runs. Target allocation:

- **Training runs** (~6–8 h wall-clock, runs in Colab background) — kicked off early, monitored periodically
- **Audio pipeline** (~4 h coding) — done while Phase A pretrain runs
- **Evaluation module** (~3 h coding) — done while Phase B + ablations run
- **README sections** (~1 h writing) — done after eval numbers land

If you have a second Google account, run the 3 ablation trainings in parallel Colab tabs. Halves the training wall-clock.

## Critical Decisions (Cursor: read this section before every session)

- **Spec is the contract.** `docs/specs/audio_pipeline.md`, `docs/specs/evaluation.md`, `docs/specs/training_runs.md` are authoritative. Interface signatures, required tests, and acceptance criteria are in there. If instinct conflicts with the spec, the spec wins.
- **Don't touch Jess-owned files.** Hands off: `src/training/*`, `src/models/hybrid.py`, `src/models/baseline.py`, `src/pipeline/run_pipeline.py`, `src/postprocess/voice_leading.py`, `src/app/*`, `configs/train.yaml`. If one of those needs a change, open an issue or ping Jess — don't edit directly.
- **Branching:** one branch per lane, named `feat/<area>-<short-desc>` (e.g. `feat/audio-pipeline`, `feat/evaluation`, `feat/ablation-runs`). Cut from `main`.
- **Commits:** natural-language messages. No `feat:`/`fix:` prefix, no AI co-author footer. Match the existing style (`git log --oneline` to see).
- **PRs:** open a draft PR as soon as you have something committed, even if it's incomplete. Lets Jess catch design mismatches early. Final PR body must include a test plan.
- **Tests:** every public function gets a test. `pytest` must stay green (currently 115/115). Add to `tests/` following the patterns in `tests/test_models.py` and `tests/test_training.py`. No test, no merge.
- **Code conventions (enforced by `.cursorrules`):**
  - `from __future__ import annotations` at the top of every Python module
  - Absolute imports: `from src.x import y` (never relative)
  - Type hints on every function. No `Any` unless truly dynamic. Use `list[int]`, `dict[str, T]`, `str | None` (3.10+ syntax, no `typing.List`).
  - `logging.getLogger(__name__)` — never `print` in `src/` code.
  - Module docstring explaining purpose; function docstrings on all public APIs.
- **Warnings discipline:** before merging, run `pytest -W error::UserWarning -W "default:enable_nested_tensor is True:UserWarning"` to confirm no new warnings introduced.
- **When in doubt, stop and ping Jess.** Don't guess at an interface. Don't invent scope beyond the spec. Don't force-merge to hit a deadline.

## Environment

- Python 3.10 (conda env `aca-adapt`). Activate with `conda activate aca-adapt` before running anything.
- Every test command in this plan assumes that env.
- Colab: T4 GPU runtime required for training. See `notebooks/03_train.ipynb` — cells are already wired.

## Tasks — all due by end of Tuesday night

Ten steps, organised by dependency + parallelisability. Step 0 is a one-time local data bootstrap. Kick off the Colab training runs first (they're wall-clock blockers), code during the training wait, finish with eval + docs.

- [ ] 🟥 **Step 0: Local data bootstrap (~15 min, one-time)**
  - [ ] 🟥 Datasets are gitignored (jaCappella has a license, and the processed `.pt` files are regenerable), so you need to build them locally before Steps 5 / 7 can run their CLIs and tests. See `SETUP.md` §4 for the canonical flow — the steps below are the short version.
  - [ ] 🟥 Confirm env is set up: `conda activate aca-adapt` (or create it per `SETUP.md` §2) and `pip install -r requirements.txt`.
  - [ ] 🟥 Accept the jaCappella license at https://huggingface.co/datasets/jaCappella/jaCappella (same HF account you'll use for Colab), then `huggingface-cli login` with a read token.
  - [ ] 🟥 `python scripts/download_data.py` — fetches jaCappella MIDI + materialises JSB Chorales into `data/raw/` (~50 MB, ~5 min).
  - [ ] 🟥 `python scripts/prepare_data.py` — tokenises + windows + augments into `data/processed/{train,val,test}.pt` (~2–5 min). Pass `--force` if you need to rebuild later.
  - [ ] 🟥 Verify: `ls data/processed/` shows `train.pt`, `val.pt`, `test.pt`. If anything errors, stop and ping Jess — don't hand-edit the data.
  - [ ] 🟥 Note: Colab training in Steps 2/4/6 downloads its own copy of the data inside the notebook; this local bootstrap is specifically for eval CLIs + unit tests on your machine.

- [ ] 🟥 **Step 1: Unblock the pipeline (~20 min)**
  - [ ] 🟥 Review PR #7 (https://github.com/jessyy2006/acapella-arranger/pull/7). Skim the 6 commits; focus on `src/training/train.py` (the main loop), `configs/train.yaml`, and `notebooks/03_train.ipynb`.
  - [ ] 🟥 If questions, comment on the PR. If approved, merge (squash) and delete the branch.
  - [ ] 🟥 After merge, pull `main` locally: `git checkout main && git pull --ff-only`.

- [ ] 🟥 **Step 2: Kick off Phase A pretrain on Colab (~2h wall-clock, background)**
  - [ ] 🟥 Open `notebooks/03_train.ipynb` in Colab. Runtime → Change runtime type → **T4 GPU**.
  - [ ] 🟥 Run cells 1–5 sequentially. Cell 2 needs your HuggingFace read token (must have accepted the jaCappella license at https://huggingface.co/datasets/jaCappella/jaCappella first).
  - [ ] 🟥 Run cell 5 (`--phase pretrain`). ~2h wall-clock — **start it before you start coding**.
  - [ ] 🟥 Keep the Colab tab visible to prevent idle-disconnect. If it disconnects, re-run all cells from top — training auto-resumes from `checkpoints/phase_a/last.pt`.
  - [ ] 🟥 When done: verify `checkpoints/phase_a/phase_a_final.pt`, `reports/phase_a_loss.csv`, `reports/phase_a_loss.png` exist on your Drive. Share the Drive folder with Jess (`william.hao@utexas.edu`) and ping final val loss + wall-clock.

- [ ] 🟥 **Step 3: Audio pipeline (~4h coding, foreground while Phase A trains)**
  - [ ] 🟥 Cut branch: `git checkout -b feat/audio-pipeline` from `main`.
  - [ ] 🟥 Read `docs/specs/audio_pipeline.md` end-to-end. This is the interface contract. Read the files the spec's "Files to read first" section lists.
  - [ ] 🟥 Create `src/pipeline/audio_to_midi.py` following the spec's function signatures exactly.
    - Stage 1: HT-Demucs call to isolate lead vocal from the mix
    - Stage 2: `torchcrepe` pitch-tracking on the isolated vocal
    - Stage 3: Quantize the pitch contour to the 16th-note grid using `src/data/vocab.py`'s `duration_to_token`
    - Return a `music21.stream.Part` OR our token format (match spec's choice)
  - [ ] 🟥 Create `tests/test_audio_pipeline.py`. At minimum: test with a synthetic sine-wave audio input (no real mp3 needed for CI); assert output shape and token types.
  - [ ] 🟥 Smoke-test on one real mp3 (any short vocal track); open the output MIDI in MuseScore to visually confirm.
  - [ ] 🟥 Confirm output plugs into Jess's upcoming `src/pipeline/run_pipeline.py` — the token format must match what the training data produced.
  - [ ] 🟥 Commit in small chunks (Demucs wrapper → CREPE wrapper → quantizer → end-to-end). Open a draft PR after the first commit, final PR ready for review when done. 115 tests still green.

- [ ] 🟥 **Step 4: Phase B fine-tune (~1h wall-clock, background — starts after Phase A)**
  - [ ] 🟥 Run cell 6 in `notebooks/03_train.ipynb` (`--phase finetune --init-from checkpoints/phase_a/best.pt`).
  - [ ] 🟥 `checkpoints/phase_b/last.pt` wins over `--init-from` on resume so disconnects don't silently revert to Phase A weights.
  - [ ] 🟥 Share `phase_b_final.pt`, loss CSV/PNG with Jess when done.

- [ ] 🟥 **Step 5: Evaluation harness (~3h coding, foreground while Phase B + ablations train)**
  - [ ] 🟥 Cut branch `feat/evaluation` from `main`.
  - [ ] 🟥 Read `docs/specs/evaluation.md`.
  - [ ] 🟥 Create `src/eval/evaluate.py`. Required metrics:
    - Per-voice token accuracy (top-1 accuracy on non-PAD positions)
    - Bar-wise accuracy (percent of bars where all 4 voices match ground truth)
    - Range compliance (percent of notes inside standard SATB vocal ranges)
    - Duration-bucket accuracy (pitch vs duration accuracy separately)
  - [ ] 🟥 Create `tests/test_evaluation.py` — synthetic logits + targets, verify each metric computes correctly including edge cases (all-PAD batch, perfect predictions, random predictions).
  - [ ] 🟥 CLI: `python -m src.eval.evaluate --checkpoint <path> --split data/processed/test.pt --report reports/eval_<run_name>.json`.

- [ ] 🟥 **Step 6: Ablation training runs (~3h wall-clock, run in parallel Colab tabs on separate Google accounts)**
  - [ ] 🟥 Run A: baseline model, same two-phase regime. Use `--model-class baseline` on both phases in the notebook.
  - [ ] 🟥 Run B: hybrid, JSB-only (no Phase B). Just skip the fine-tune cell. Use a distinct `run_name` so you don't overwrite Phase A's `checkpoints/phase_a/`.
  - [ ] 🟥 Run C: hybrid, jaCappella-only (no Phase A). Run `--phase finetune` without `--init-from` — fresh init on jaCappella only.
  - [ ] 🟥 Collect all 5 runs' checkpoints + loss curves + val losses into the Drive folder. Coordinate with Jess before running if config tweaks are needed for run C.

- [ ] 🟥 **Step 7: Ablation analysis + plots (~1h)**
  - [ ] 🟥 Create `src/eval/ablation.py` — loads the 5 eval JSONs and produces a comparison table.
  - [ ] 🟥 Run `evaluate.py` on all 5 checkpoints against the same held-out test set. Produces `reports/eval_<run_name>.json` per run.
  - [ ] 🟥 Generate comparison plots (matplotlib, Agg backend): val-loss curves overlaid across the 5 runs; bar chart of per-metric comparison. Save to `reports/plots/`.
  - [ ] 🟥 Commit plots as PNGs directly to the repo (spec allows this — they're small and the rubric wants them visible).

- [ ] 🟥 **Step 8: Per-run markdown notes (~30 min)**
  - [ ] 🟥 For each of the 5 training runs, write a 1-paragraph writeup at `reports/run_notes/<run_name>.md`: final val loss, best val loss, wall-clock time, anomalies, any config tweaks from default. Grader + Jess both need these to interpret the ablation numbers.
  - [ ] 🟥 Add a top-level `reports/README.md` indexing the run notes + plots so a grader can navigate.

- [ ] 🟥 **Step 9: README first-draft Evaluation + Methodology sections (~1h)**
  - [ ] 🟥 Add "Evaluation" and "Methodology" sections to `README.md`. Reference the plots + numbers from Steps 7–8. Evidence-driven prose: every claim about model performance cites a specific metric + plot.
  - [ ] 🟥 Do not touch other README sections — those are Jess's lane.

## Post-deadline (slips past Tuesday night — do Wed/Thu)

- **Technical walkthrough video (~5 min)** — script: problem framing (30s) / pipeline overview (60s) / model architecture (90s) / training recipe (60s) / results walkthrough (60s). Export as `videos/technical_walkthrough.mp4`. Target: Thu/Fri.
- **Gradescope submission** — Sun 2026-04-26 by 11:59 pm. Both partners fill out the "Individual Contributions" section of the README (rubric-required).

## Blocker-escalation rules

- **If Colab consistently OOMs**: halve `batch_size` in `configs/train.yaml` (16 → 8). Document the change in your run notes.
- **If training loss diverges / hits NaN**: stop the run. Grab the traceback + the last 10 epochs of CSV. Ping Jess; don't try to fix the training loop.
- **If a spec is unclear**: read the spec again, read the referenced files, check if the answer is obvious from existing code patterns. If still unclear, post a targeted question in the issue tracker or DM Jess. Don't guess and ship.
- **If you break an existing test**: stop, don't force. Either your change is wrong, or the test was wrong; figure out which before proceeding.

## Handoff checkpoints (ping Jess at these milestones)

1. ✋ **PR #7 merged** — Jess needs to know so she can start the voice-leading post-processor + `run_pipeline.py`.
2. ✋ **Phase A final.pt on Drive** — unblocks Phase B kickoff AND gives Jess something to smoke-test the deployed app against.
3. ✋ **Phase B final.pt on Drive** — production weights for the app.
4. ✋ **Ablation numbers ready** — Jess needs these for the final README.
5. ✋ **All 5 evals done** — triggers Jess's final README polish pass.
