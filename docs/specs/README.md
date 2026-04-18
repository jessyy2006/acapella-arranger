# Lane Specs

Per-lane contracts for the work items under your ownership. Each spec is a standalone contract — if you build against one and its tests pass, the integration with the rest of the project works by construction.

## Your specs

| Spec | Deliverable | Rubric value |
|---|---|---|
| [`baseline_model.md`](./baseline_model.md) | `src/models/baseline.py` — pure-Transformer seq2seq for ablation comparison | ~3 pts |
| [`audio_pipeline.md`](./audio_pipeline.md) | `src/pipeline/audio_to_midi.py` — audio file → tokenised lead melody | feeds multi-stage pipeline (~7 pts) + cross-modal (~7 pts) |
| [`evaluation.md`](./evaluation.md) | `src/eval/*` + `reports/*.md` — metrics + ablation + error analysis | ~3 + 5 + 7 + 7 = ~22 pts |
| [`training_runs.md`](./training_runs.md) | Colab Phase A + Phase B runs; checkpoints + loss curves | enables ~25 pts worth of claimed ML items |

## How a spec file is organised

Every spec has the same sections:

- **Goal** — one sentence, the thing you're shipping.
- **Rubric justification** — which rubric items this closes and the point value, so you know what's at stake.
- **Interface contract** — exact function signatures, input/output types and shapes. Binding.
- **Dependencies to import** — which existing modules you consume. Do not re-implement something that exists.
- **Required tests** — cases that must pass before PR. Additional tests welcome.
- **Design freedom** — the implementation decisions you get to make, with a recommended default.
- **Gotchas** — common pitfalls I've already thought about.
- **Files to read first** — the existing source code you should skim before writing.
- **Acceptance criteria** — how Jess verifies the PR is done.

## Workflow for each lane

```
1. Read the spec top-to-bottom.
2. Skim the "Files to read first" list in Cursor.
3. Cut a branch:   git checkout -b feat/<lane>-<desc>
4. Stub the file with the required interface + docstrings.
5. Write the required tests FIRST — they'll fail, that's fine.
6. Implement until tests pass.
7. Run the full pytest suite — nothing else should break.
8. Open a draft PR to main. Don't wait until it's "done".
9. Iterate with Jess's review.
10. Merge when approved.
```

## What counts as "done"

- All required tests in the spec pass.
- `pytest` is green across the whole repo.
- Acceptance criteria at the bottom of the spec are met.
- PR description links the spec and notes any deviations.
- No `print`, no `TODO`, no commented-out debug code in `src/`.

## When the spec is wrong

Specs are a best-effort contract, not infallible. If you hit something that doesn't make sense:

- **Design choice**: pick the reasonable one and document it in the PR description ("I chose X over Y because Z"). Jess will call it out if she disagrees.
- **Missing information**: open a draft PR asking, rather than guessing silently.
- **Actual contradiction**: ping Jess directly. We'd rather fix the spec than build against the wrong one.

## Branching + PR reminders

- Cut from `main`, not from each other's branches.
- One branch per lane.
- Commit style: natural language, no `feat:` prefix, no AI co-author footer. See `.cursorrules` for examples.
- PRs can be draft from day 1. Early feedback beats late rework.
