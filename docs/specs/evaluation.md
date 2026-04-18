# Spec — Evaluation + Ablation Study

## Goal

Build the evaluation harness that measures model quality and answers the question "did the design choices actually matter?". Three deliverables:

1. `src/eval/metrics.py` — reusable metric functions that operate on model outputs + targets.
2. `src/eval/evaluate.py` — CLI that loads a checkpoint, runs it against the test split, computes every metric, and writes a human-readable report.
3. `src/eval/ablation.py` — CLI that runs multiple trained configurations through `evaluate.py` and produces a comparison table + plots.

Plus a set of markdown/PNG reports under `reports/` that the grader reads.

## Rubric justification

This is the **highest-leverage** spec you own — it directly claims ~22 rubric pts:

- **≥3 distinct evaluation metrics** — 3 pts
- **Both qualitative + quantitative evaluation with discussion** — 5 pts
- **Ablation study varying ≥2 design choices** — 7 pts
- **Error analysis with failure case visualisation** — 7 pts

Evidence is required for every one of these claims. "We have metrics" isn't enough — we need numerical tables + plots + prose interpretation. The reports under `reports/` are the rubric evidence.

## Interface contracts

### `src/eval/metrics.py`

Pure functions, no side effects. Each takes model outputs and returns a scalar or dict.

```python
def next_token_accuracy(
    logits: Tensor,          # (B, L, V)
    targets: Tensor,         # (B, L)
    pad_idx: int = PAD,
) -> float:
    """Fraction of non-PAD positions where argmax(logits) == target."""

def per_voice_accuracy(
    logits_by_voice: dict[str, Tensor],
    targets_by_voice: dict[str, Tensor],
    pad_idx: int = PAD,
) -> dict[str, float]:
    """Token accuracy for each of s/a/t/b separately."""

def bar_accuracy(
    logits_by_voice: dict[str, Tensor],
    targets_by_voice: dict[str, Tensor],
    bar_token: int = BAR,
    pad_idx: int = PAD,
) -> float:
    """Fraction of bars where ALL FOUR voices are predicted exactly right."""

def pitch_range_compliance(
    predictions_by_voice: dict[str, list[int]],  # decoded token lists
) -> dict[str, float]:
    """Fraction of generated pitches within each voice's observed training range
       (see Day 1 exploration: S=[55,81], A=[50,76], T=[47,74], B=[31,64])."""

def duration_bucket_accuracy(
    logits_by_voice: dict[str, Tensor],
    targets_by_voice: dict[str, Tensor],
) -> float:
    """Token accuracy restricted to duration tokens only — does the model get
       rhythm right independent of pitch?"""

def voice_crossing_rate(
    predictions_by_voice: dict[str, list[int]],
) -> float:
    """Fraction of aligned time steps where the 'higher' voice is NOT above
       the 'lower' voice — i.e. bass rises above tenor, tenor above alto, etc.
       Lower is better. Measures whether the arrangement respects SATB
       vertical ordering."""
```

Implement at least the five bolded in the rubric list. More is fine; document which are core vs exploratory.

### `src/eval/evaluate.py`

```python
def evaluate_checkpoint(
    checkpoint_path: Path,
    test_dataset_path: Path,
    *,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, float]:
    """Load model + test split, run inference, compute all metrics. Returns
       a flat dict that maps metric_name -> scalar for the final report."""

def main() -> int:
    """argparse CLI entrypoint. Writes reports/<checkpoint_name>_metrics.md."""
```

### `src/eval/ablation.py`

```python
def run_ablation(
    checkpoint_paths: dict[str, Path],  # {"hybrid": ..., "baseline": ..., ...}
    test_dataset_path: Path,
) -> pd.DataFrame:
    """Run every checkpoint through evaluate_checkpoint. Return a DataFrame
       with one row per checkpoint, one column per metric."""

def plot_ablation_bar_chart(df: pd.DataFrame, out_path: Path) -> None:
    """One subplot per metric, bars for each variant. Saves PNG."""

def main() -> int:
    """argparse CLI. Writes reports/ablation.md + reports/ablation_<axis>.png."""
```

## Ablation axes (required)

Rubric requires ≥2 axes. Run all three:

1. **Model architecture**: hybrid (Jess) vs baseline (you). Same training data, same hyperparameters, same training budget. Only `nn.Module` class differs.
2. **Voice-leading post-process**: on vs off. Same trained model, toggle the post-processor in the inference path. Metrics to check: voice-crossing rate, range compliance. Expected: post-process improves both.
3. **Pretraining regime**: (a) JSB-only, (b) jaCappella-only, (c) JSB pretrain + jaCappella fine-tune (default). Metrics on jaCappella test set. Expected: full regime wins.

Each axis is one section of `reports/ablation.md`. Prose discussion with hypothesis + observation + interpretation.

## Required reports

### `reports/metrics.md`

A single table of all metrics on the final (fine-tuned hybrid) checkpoint, evaluated against the test split. Short prose paragraph interpreting the headline numbers.

### `reports/ablation.md`

Three sections (one per axis). Each section:

- **What varied**: which knob was changed
- **Hypothesis**: what we expected to happen, and why
- **Results**: numerical table + link to the plot
- **Interpretation**: 2–3 sentences on whether the hypothesis held, and what we conclude

### `reports/error_analysis.md`

Pick 3–5 representative failure cases from the test set. For each:

- The input lead (first few bars on a staff, or a described MIDI snippet)
- The generated SATB (staff image or described snippet)
- What's wrong (parallel fifths? out-of-range note? rhythmic collision?)
- Why it likely happened (insufficient training data? post-process limitation? tokenizer quantisation?)

Use `music21.Stream.show("musicxml.png")` or matplotlib piano-roll plots for the visualisations. Save PNGs under `reports/plots/`.

### `reports/plots/*.png`

- Training loss curves (Phase A, Phase B — from the training runs)
- Ablation bar charts (one per axis)
- Failure-case piano rolls

## Required tests

New file `tests/test_eval.py`. Goals are lighter than the metrics themselves — we're verifying the harness works, not that metrics are correct on real model output (that's what the reports are for).

1. `test_metrics_on_known_input` — hand-build a tiny logits tensor where you know the expected accuracy (e.g., 100% correct → 1.0; all wrong → 0.0; half right → 0.5). One test per metric.
2. `test_pad_positions_excluded` — metrics ignore PAD. Build a batch where the non-pad positions are all correct but the pad positions are all wrong; accuracy should still be 1.0.
3. `test_evaluate_checkpoint_runs_e2e` — save a dummy `SATBHybrid` checkpoint to `tmp_path`, run `evaluate_checkpoint` on a 2-example test set, assert the return dict has all expected keys. Use the existing `TestPersistence` test in `test_dataset.py` as a template.
4. `test_ablation_handles_missing_checkpoint` — if a checkpoint path doesn't exist, error with a clear message (not a cryptic pickle error).

## Design freedom

1. **Report format** — markdown is required (graders read it), but you can add a Jupyter notebook `notebooks/04_eval.ipynb` if you want a live exploratory version. Optional.
2. **Plot style** — matplotlib with default seaborn palette is fine. Your call on colour / layout, as long as plots are readable and labeled.
3. **Prose voice** — the interpretation sections should be technically honest (if a hypothesis fails, say so). This is a grading rubric-aware document: interpretation matters more than pretty numbers.
4. **Which checkpoints to evaluate** — at minimum the three ablation axes above (that's 3 × ~2 = 6 configurations). More is fine; don't stretch past 10 (diminishing returns and Colab time).

## Gotchas

- **Don't evaluate on training data** — the leakage tests in `tests/test_dataset.py` guarantee train/val/test are disjoint at the song level, but your evaluation code needs to use the *test* split specifically. `torch.load(data/processed/test.pt, weights_only=False)` — or call `src.data.loaders.load_dataset`.
- **Inference vs teacher-forcing evaluation** — next-token accuracy with teacher forcing is easier than free-running generation. Both are useful, but be explicit in the report which you're measuring. Recommend starting with teacher-forced (simpler, matches training loss) and adding a free-running "bar completion" evaluation as a qualitative case study.
- **Per-voice ranges from Day 1** (see `docs/PRD.md` §2 or the project plan): S=[55,81], A=[50,76], T=[47,74], B=[31,64]. These are the training-distribution ranges; use them for `pitch_range_compliance`.
- **`voice_crossing_rate`** needs aligned predictions — but each voice has its own length after decoding. Either (a) use teacher-forced predictions where lengths match the targets, or (b) align by time (re-emit predictions per 16th-note position and compare at each position).
- **Bar-wise accuracy is punishing** — a single wrong token in any voice fails the bar. Expect this number to be much lower than token accuracy. That's fine; the contrast is informative.
- **Determinism** — seed torch and numpy before evaluation. Otherwise two runs on the same checkpoint + test set produce slightly different numbers due to cuDNN non-determinism.
- **Checkpoint format** — Jess's training loop will save via `torch.save(model.state_dict(), path)`. Your `evaluate.py` reconstructs the model (`SATBHybrid()`) then `model.load_state_dict(torch.load(path))`. Coordinate with Jess if her format differs.

## Files to read first

1. **`src/models/hybrid.py`** — forward-pass contract. Your evaluation code will call `model(batch)` exactly as the training loop does.
2. **`src/data/loaders.py`** — `load_dataset`, `collate_satb`, `make_dataloaders`. You'll use all three.
3. **`src/data/vocab.py`** — PAD, BAR, REST, `is_pitch_token`, `token_to_pitch`. Your metrics will inspect raw tokens.
4. **`tests/test_models.py`** — shows how to build a synthetic batch for unit-testing metrics.
5. **Day 1 exploration notebook** (`notebooks/01_data_exploration.ipynb`, cells that compute per-voice ranges) — source for range compliance metric.
6. **Course rubric HTML** (`final_project_handout.html`) — specifically the "Evaluation" section to confirm metric-count requirement.

## Acceptance criteria

- [ ] All 5+ metric functions in `metrics.py` with unit tests.
- [ ] `evaluate.py` produces `reports/metrics.md` from a checkpoint + test set via CLI.
- [ ] `ablation.py` produces `reports/ablation.md` + 3 PNG bar charts.
- [ ] `reports/error_analysis.md` with 3–5 failure cases, each with viz + diagnosis.
- [ ] Training loss curves saved to `reports/plots/` (source: the training-runs task; you just plot them).
- [ ] Full `pytest` suite stays green.
- [ ] PR description summarises: headline metrics on the test set + the most interesting ablation finding.

## Out of scope

- **Training the models** — training runs are a separate task. You evaluate what's given to you.
- **Human listening tests** — "human-evaluated MOS" is a rubric item (10 pts) but out of scope for this timeline. Don't start one.
- **Publishing to a leaderboard** — there's no "SATB arrangement benchmark" to compare against. Beating our own baseline IS the story.
