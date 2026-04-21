# Soham — Next Steps (post PR #9)

**Tuesday-night deadline unchanged.** Priority-ordered, actionable checklist — tick boxes as you go. Every item here is either rubric-critical or blocks a rubric-critical item.

## TL;DR

Your 2 hours of Phase A + Phase B training are **fine**. The bad-looking metrics in `reports/` came from broken *evaluation* code, not broken training. PR #9 ([fix/evaluation-review](https://github.com/jessyy2006/acapella-arranger/pull/9)) fixes 4 metric bugs + the hparam-mismatch that made every reported number wrong, and adds the missing `tests/test_eval.py` the spec required.

Three things to do, in order:

1. **Verify the fix works** against your real checkpoints (~30 min)
2. **Run real ablation axes + error analysis** (~4h, mostly Colab wall-clock)
3. **Write per-run notes + flesh out `reports/ablation.md`** (~1h)

## Priority 1 — Verify PR #9 against your trained checkpoints (~30 min)

Do this **before** you merge the PR. If the fix works, the PR is safe to land; if it doesn't, we have a second bug and you shouldn't merge yet.

### 1a. Pull the fix branch in Colab

Colab is the fastest path because your checkpoints are already on the Colab-mounted Drive. Open `notebooks/03_train.ipynb` (or any Colab notebook with the repo cloned) and run:

```python
%cd /content/acapella-arranger
!git fetch origin
!git checkout fix/evaluation-review
!git pull --ff-only
```

If you prefer local: `git fetch && git checkout fix/evaluation-review` in your clone. You'll also need to run the local data bootstrap from Step 0 of `PARTNER_EXECUTION_PLAN.md` if you haven't already.

### 1b. Run the fixed eval against Phase A

```bash
python -m src.eval.evaluate \
  --checkpoint /content/drive/MyDrive/<your-folder>/phase_a_final.pt \
  --split data/processed/test.pt \
  --model-class hybrid \
  --out-json reports/phase_a_eval_v2.json \
  --out-md reports/phase_a_eval_v2.md
```

Swap the `--checkpoint` path for wherever `phase_a_final.pt` lives on your Drive.

**Note on hparams:** the fixed `evaluate.py` auto-reads model hparams from `configs/train.yaml` if no sidecar config exists, so as long as you trained with the default config (you did — the `d_model=256` in the CSV matches the yaml), no extra setup is needed. If you ever train a variant with tweaked hparams, drop a sibling `<checkpoint-stem>.config.json` next to the `.pt` with the right values.

### 1c. Compare the new numbers against the training CSV

Open `reports/phase_a_eval_v2.md` and check `acc_s`. It should land in the **0.55–0.65** range.

- **Training CSV reference** (`reports/phase_a_loss.csv`, epoch 24): `val_acc_s = 0.6265`, `val_acc_a = 0.4920`, `val_acc_t = 0.5427`, `val_acc_b = 0.4526`.
- **Expect within ~5 points** on each voice — eval is on the test split, training CSV is on the val split, so they're not identical but should be close.

Expected results:

| outcome | what it means | action |
|---|---|---|
| `acc_s ≈ 0.6`, all voices close to training CSV | Fix works — `_build_model` was the bug | Proceed to 1d |
| `acc_s` still ~0.28, or way off | Second bug — stop and ping Jess with the output JSON | Don't merge |
| Error on load (shape mismatch, etc.) | Likely a checkpoint-format issue | Ping Jess with the full traceback |

Also check `bar_acc` — it should now be **non-zero** (previously 0.0000 everywhere). A low-but-positive number (e.g. 0.01–0.05) is correct; per-bar exact match is punishing.

### 1d. Merge PR #9 into `feat/evaluation`

If Priority 1c looks good:

```bash
gh pr merge 9 --squash --delete-branch=false
```

Or merge via the GitHub UI. Do **not** delete `fix/evaluation-review` until `feat/evaluation` is merged to `main` — the stacked branch history is useful if anything needs reverting.

Then update your local checkout:

```bash
git checkout feat/evaluation
git pull --ff-only
```

### 1e. Re-run Phase B eval and overwrite the old reports

With the fixed harness, regenerate `reports/phase_a_final_metrics.{json,md}` and `reports/phase_b_final_metrics.{json,md}` so the README can cite real numbers:

```bash
python -m src.eval.evaluate --checkpoint <phase_a_final.pt> --split data/processed/test.pt \
  --out-json reports/phase_a_final_metrics.json --out-md reports/phase_a_final_metrics.md
python -m src.eval.evaluate --checkpoint <phase_b_final.pt> --split data/processed/test.pt \
  --out-json reports/phase_b_final_metrics.json --out-md reports/phase_b_final_metrics.md
```

Commit them to `feat/evaluation` with a message like `regenerate phase a/b eval with fixed harness`.

## Priority 2 — Real ablation axes (~3h wall-clock + 1h code)

Current `reports/ablation.md` compares "phase_a vs phase_b" — those are **two checkpoints of the same run**, not an ablation. The rubric wants ≥2 *design-choice* axes. Per `docs/specs/evaluation.md`, the three required axes are:

1. **Architecture** — hybrid vs baseline (same training recipe)
2. **Voice-leading post-process** — on vs off (same model)
3. **Pretraining regime** — JSB-only / jaCappella-only / combined

You have the combined case already (Phase A + Phase B). The cheapest missing axis is **architecture** — train the baseline on the same two-phase regime. Do this first.

### 2a. Train the baseline (~2h wall-clock Colab, background)

In `notebooks/03_train.ipynb`, run the Phase A + Phase B cells with `--model-class baseline` on both. Use a distinct `run_name` in the CLI flags so the baseline checkpoints don't overwrite your hybrid ones. Reference: `docs/specs/training_runs.md`.

If you have a second Google account, run this in parallel with step 2b.

### 2b. Train the jaCappella-only variant (~1h Colab, optional if time-tight)

Phase B only, no `--init-from`:

```bash
python -m src.training.train --config configs/train.yaml --phase finetune --run-name jacap_only
```

Skip this if the architecture axis + error analysis push you past the deadline. Two axes clear the rubric threshold; three is nicer but not required.

### 2c. Run the ablation CLI once all checkpoints exist

```bash
python -m src.eval.ablation \
  --split data/processed/test.pt \
  --checkpoint phase_a <path/to/phase_a_final.pt> \
  --checkpoint phase_b <path/to/phase_b_final.pt> \
  --checkpoint baseline_phase_b <path/to/baseline_phase_b.pt> \
  --checkpoint jacap_only <path/to/jacap_only_final.pt> \
  --out-md reports/ablation.md \
  --out-png reports/plots/ablation.png
```

### 2d. Rewrite `reports/ablation.md` prose

The CLI produces a table. You need to add, per spec, **three sections** (one per axis) with:

- **What varied** — which knob changed
- **Hypothesis** — what you expected and why
- **Results** — numerical table + link to the plot
- **Interpretation** — 2–3 sentences on whether the hypothesis held

Template is in `docs/specs/evaluation.md` §"Required reports". Keep it honest — if an axis didn't move the numbers much, say so.

## Priority 3 — Error analysis (~1h, ~7 rubric pts)

Create `reports/error_analysis.md` with 3–5 representative failure cases pulled from the test set. For each case:

- The input lead (first few bars — piano roll PNG or described MIDI)
- The generated SATB (piano roll PNG)
- **What's wrong** (parallel fifths? out-of-range note? rhythmic collision? voice crossing?)
- **Why it likely happened** (insufficient training data? tokenizer quantisation? lack of voice-leading post-process?)

Use `music21.Stream.show("musicxml.png")` or `matplotlib` piano-roll plots. Save PNGs to `reports/plots/`. Spec: `docs/specs/evaluation.md` §"`reports/error_analysis.md`".

**How to find failure cases:** after running ablation, look at samples where `bar_accuracy` is low or `voice_crossing_rate` is high for a specific sample. Eyeball 10 candidates in a notebook, pick the 3–5 most instructive (ideally one of each failure mode, not all the same bug).

## Priority 4 — Per-run notes + README index (~30 min)

### 4a. `reports/run_notes/<run_name>.md` for each of the 5 runs

One paragraph each: final val loss, best val loss, wall-clock, any config tweaks from default, anomalies (e.g. Phase B overfit — val loss bottomed epoch 4 then drifted up).

Runs to document: `phase_a`, `phase_b`, `baseline_phase_a`, `baseline_phase_b`, `jacap_only` (or whatever subset you end up with).

### 4b. `reports/README.md` index

Top-level index so a grader can navigate. List each `run_notes/*.md`, each plot in `plots/`, and `ablation.md`, `error_analysis.md`, `metrics.md`. One-line description per entry.

## Priority 5 — README evaluation + methodology sections (~1h, Step 9 of partner plan)

Only after Priorities 1–4 land. Evidence-driven prose that cites the plots + numbers from Priority 2 and 3. Don't touch other README sections — those are Jess's.

## Blockers / if things go wrong

- **P1c shows `acc_s` still ~0.28 or wildly off training CSV**: don't merge PR #9. Post the full JSON output + the command you ran in the partner thread. There's a second bug and we need to find it.
- **Baseline training OOMs on Colab**: halve `batch_size` in `configs/train.yaml` (16 → 8) and re-run. Document the change in that run's notes.
- **Training loss diverges / hits NaN**: stop the run, grab the last 10 epochs of CSV + traceback, ping Jess. Don't try to fix the training loop — that's Jess's lane.
- **Out of Colab GPU time**: skip Priority 2b (jaCappella-only). Two axes (architecture + pretraining-via-A-vs-B) is enough for the rubric.
- **Running out of Tuesday-night time**: drop Priority 2b first, then Priority 4 (run notes can be terse), then reduce Priority 3 to 3 cases instead of 5.

## Final merge to `main`

Once Priorities 1–4 are on `feat/evaluation`, open the `feat/evaluation → main` PR. Jess reviews + merges. Only after that do you start Priority 5 (README sections land in a separate PR on top of main).

## Contact

Ping Jess in the partner thread at each handoff checkpoint:

1. PR #9 verified + merged (Priority 1d done) — unblocks Jess from noting the numbers.
2. Baseline checkpoint on Drive — Jess may want to smoke-test the deployed app against it.
3. `feat/evaluation` merged to `main` — triggers Jess's final README polish pass.

Submission deadline: **Sun 2026-04-26, 11:59 pm**. Tuesday night is the code/eval deadline; buffer days after that are Jess's integration + final polish only.
