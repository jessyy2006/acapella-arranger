# Deploying Aca-Adapt to Hugging Face Spaces

Step-by-step. Assumes you already have a working local setup per [SETUP.md](SETUP.md) and a trained checkpoint at `checkpoints/phase_b/phase_b_final.pt`.

The deploy has two pieces — a **model repo** that hosts the 90 MB `phase_b_final.pt` (too big for GitHub), and a **Space** that runs the Streamlit app and pulls the checkpoint from the model repo at boot.

## 1. One-time HF setup

1. Sign up at <https://huggingface.co> if you don't have an account.
2. Generate a write-scope token: <https://huggingface.co/settings/tokens> → New token → Role: Write. Save the string; you'll paste it in a few places.
3. Authenticate locally so the upload script can push for you:

   ```bash
   conda activate aca-adapt
   huggingface-cli login
   # Paste the token when prompted.
   ```

## 2. Upload the checkpoint to HF Hub

Pick a repo name — convention is `<your-hf-username>/aca-adapt-model`.

```bash
python scripts/upload_checkpoint_to_hf.py \
  --checkpoint checkpoints/phase_b/phase_b_final.pt \
  --repo-id <your-hf-username>/aca-adapt-model
```

This creates the repo if it doesn't exist and uploads `phase_b_final.pt` to it. Re-run the same command after any retrain to overwrite.

Add `--private` if you don't want the weights publicly visible. A private repo still works with the Space but you'll have to supply an `HF_TOKEN` secret on the Space side so it can download.

## 3. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Pick a name (e.g. `aca-adapt`), SDK: **Streamlit**, hardware: **CPU basic (free)**.
3. Create. HF gives you an empty Space repo.

## 4. Point the Space at your model repo

In the new Space: **Settings → Variables and secrets** → add:

| Name | Value |
|---|---|
| `ACA_ADAPT_HF_REPO_ID` | `<your-hf-username>/aca-adapt-model` |

That's the only required variable. Optional ones:

| Name | When to set |
|---|---|
| `ACA_ADAPT_HF_FILENAME` | Only if you uploaded under a non-default filename. |
| `HF_TOKEN` | Only if the model repo is private. Generate a **read-scope** token and store it as a *Secret* (not a plain Variable). |

## 5. Push the app code into the Space

Two options — pick one.

### Option A — GitHub Action (recommended; push-to-deploy loop)

Uses the workflow already at `.github/workflows/sync-to-hf-space.yml`.

1. On GitHub, go to this repo → **Settings → Secrets and variables → Actions → New repository secret**, and add:

   | Name | Value |
   |---|---|
   | `HF_TOKEN` | The write-scope token from step 1. |
   | `HF_USERNAME` | Your HF username. |
   | `HF_SPACE_NAME` | The Space name from step 3 (e.g. `aca-adapt`). |

2. Push any commit to `main`. The workflow fires, force-pushes the repo contents into `https://huggingface.co/spaces/<user>/<space-name>`, and HF rebuilds the Space in 1–2 minutes.

From here, every `git push` you do in this repo → Space updates automatically. Same iteration loop as Streamlit Community Cloud.

### Option B — Manual one-shot push

Useful if you don't want the GitHub Action.

```bash
# In the project root:
git remote add hf https://huggingface.co/spaces/<your-hf-username>/<space-name>
git push hf main
# Paste your HF username + write token when prompted (or store them
# in ~/.huggingface/token beforehand).
```

Re-run `git push hf main` any time you want to redeploy.

## 6. Verify

1. Open your Space URL (`https://huggingface.co/spaces/<user>/<space-name>`).
2. First boot takes 2–4 minutes — HF builds the container + installs `requirements.txt`.
3. Upload the smoke clip (or any 30 s – 3 min audio file).
4. Click Generate. First generation also takes longer than local because Demucs weights (~320 MB) download on first run.
5. Download the MIDI.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Space build fails on `pip install` | Check the build logs. A common issue is `torch` pulling a CUDA wheel on the CPU basic hardware — if this happens, pin `torch` to the CPU wheel in `requirements.txt` (e.g. `torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`). |
| App loads but errors with "Model checkpoint not found" | `ACA_ADAPT_HF_REPO_ID` variable on the Space isn't set, or the model repo exists but the filename doesn't match. Check step 4. |
| 401 on checkpoint download | The model repo is private and you didn't add `HF_TOKEN` as a Space secret. Either make the repo public or add the token. |
| Out-of-memory during generation | CPU basic free tier is 16 GB — this app shouldn't OOM. If it does, switch `htdemucs_ft` → `htdemucs` in `src/pipeline/audio_to_midi.py:633` (cuts Demucs weight footprint by 4×). |
| App reruns every edit in advanced options | That's Streamlit's default behavior. The generated MIDI is cached in `st.session_state` keyed on file hash + options, so re-clicking Generate on the same inputs is instant. |
