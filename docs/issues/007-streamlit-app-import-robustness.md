# 007 — Streamlit app crashes with "No module named 'src'" / "No module named 'librosa'"

- **Type**: bug
- **Priority**: high (app is unusable for anyone who doesn't know the exact launch incantation — blocks the deployed-web-app rubric item)
- **Effort**: small (both paths are one-liner fixes)
- **Found in**: first user-facing exercise of the Streamlit app after `9de46f5`
- **Commit at time of report**: `9de46f5`
- **Status**: not started

## TL;DR

`streamlit run src/app/main.py` crashes depending on *how* it was launched. Two observed failure modes, same underlying class of bug (the app silently assumes a specific Python + working directory):

1. On upload → `Couldn't read that audio file: No module named 'librosa'`
2. On clicking Generate → `Unexpected error: No module named 'src'`

Both errors surface *inside* the Streamlit red box with no indication that the user's shell is the problem. The app should either make these cases impossible or surface a clear "your environment is misconfigured" message at startup.

## Current behaviour

**Failure (1)** — `_probe_audio` does `import librosa` lazily inside the function. If the `streamlit` binary on PATH is from a different environment than `aca-adapt` (e.g. user launched from `base` conda env or from a shell without `conda activate aca-adapt`), librosa isn't importable and the file probe raises. The exception is caught at `main.py:main()` and surfaced as `"Couldn't read that audio file: {exc}"` — factually true but useless for diagnosis.

**Failure (2)** — `_run_arrangement` imports `from src.pipeline.run_pipeline import run_pipeline` at call time. Streamlit adds the directory containing the script (`src/app/`) to `sys.path`, **not** the project root. So `src.*` is not importable. Any user who runs `streamlit run src/app/main.py` from the project root hits this the moment they click Generate.

## Expected behaviour

- App boots successfully regardless of whether the user remembered to `conda activate aca-adapt`.
- If a required dependency is genuinely missing, surface a clear "environment issue — librosa is not installed in this Python (`{sys.executable}`); run `pip install -r requirements.txt` or activate the `aca-adapt` env" message at startup, not buried in a per-upload exception.
- `src.*` imports work without any manual `PYTHONPATH` tweaking.

## Candidate root causes / fix options

**For (2) — the `src` import error:**

1. Add a `sys.path.insert(0, str(_PROJECT_ROOT))` at the top of `src/app/main.py`, before any `src.*` imports. One line. Makes the app work from any CWD.
2. *Or* install the project as an editable package (`pip install -e .`) — requires adding a `pyproject.toml` or `setup.py`. Cleaner long-term but more scope.

**For (1) — the librosa / env mismatch:**

1. Do an eager `import librosa, torch, torchcrepe, demucs, mido` at module top, so any missing dep throws at **startup** with a full stack trace (visible in the terminal that launched `streamlit`) instead of during a later user action.
2. *Or* add a preflight check that catches `ImportError` at startup and renders an `st.error` with actionable remediation (activate env, install requirements, point `ACA_ADAPT_PYTHON` at the right interpreter).
3. Update `SETUP.md` and the README's "Quick Start" to call out the env activation step explicitly. Low-tech but high-leverage — most users reach for the README first.

## Acceptance criteria

1. Fresh checkout + `conda activate aca-adapt` + `streamlit run src/app/main.py` from the project root produces a working app with no import errors.
2. Launching with the wrong Python (`/usr/bin/python3` or `base` env) fails fast at startup with a message that names the missing module and the expected env.
3. `src.*` imports work from any CWD.

## Relevant files

- `src/app/main.py:30–45` — module-level imports and `_PROJECT_ROOT` computation (add `sys.path.insert` here).
- `src/app/main.py:80–95` — `_probe_audio` (the lazy `import librosa` surfaces failure 1).
- `src/app/main.py:135–145` — `_run_arrangement` (the lazy `from src.pipeline...` surfaces failure 2).
- `SETUP.md` — may need a "before running the app" reminder.
- `README.md` — "Quick Start" already mentions `source .venv/bin/activate` but not `conda activate aca-adapt`; verify this matches the actual supported setup.

## Risks / notes

- The `sys.path` insert is safe when applied *before* any other imports that depend on it. Putting it at the top of the module (after stdlib imports, before any `src.*` import) avoids subtle ordering bugs.
- Eager-importing the heavy deps at module load will slow the app's cold start by a few seconds (Demucs imports torch, torch imports CUDA stuff). This is the right tradeoff — better to wait at startup than fail midway through a user action.
- Adding `pyproject.toml` / `pip install -e .` is the long-term correct fix but touches more surface area and shouldn't be bundled with this ticket.
