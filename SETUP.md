# Setup Guide

Step-by-step installation and usage for Aca-Adapt.

## Prerequisites

- **Python 3.10 or 3.11.** Python 3.12 and 3.13 are NOT supported — several audio ML libraries (Demucs, TensorFlow-adjacent stuff) lag behind the latest CPython.
- **ffmpeg** (required by Demucs and yt-dlp)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - Windows: https://ffmpeg.org/download.html (add to PATH)
- **git**
- (Optional) CUDA-capable GPU for local training. CPU is fine for inference; training is recommended on Google Colab free T4.

## 1. Clone the repository

```bash
git clone <repo-url>
cd "Acapella Arranger"
```

## 2. Create an isolated Python environment

Pick ONE of the two paths below.

### Path A — Conda (recommended if you have Anaconda / Miniconda installed)

```bash
conda create -n aca-adapt python=3.10 -y
conda activate aca-adapt
```

### Path B — venv (if you have a native Python 3.10 or 3.11 install)

```bash
# Verify you have 3.10 or 3.11
python3.10 --version   # or python3.11 --version

python3.10 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows PowerShell
```

If `python3.10` is not found, install it with `brew install python@3.10` (macOS) or use Path A.

## 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- Pitch tracking uses `torchcrepe` (PyTorch port of CREPE) — no TensorFlow dependency, so installs are fast and stable.
- On Apple Silicon, PyTorch uses MPS automatically where available.
- First inference downloads HT-Demucs weights (~2 GB) — subsequent runs use the cached weights.

## 4. Download datasets

```bash
python scripts/download_data.py
```

This will:
- Fetch jaCappella MIDI from HuggingFace → `data/raw/jacappella/`
- Materialize JSB Chorales via `music21.corpus.chorales` → `data/raw/jsb_chorales/`

Expected disk use: ~50 MB for MIDI data.

## 5. Running the pipeline

### Inference on a single song

```bash
python -m src.pipeline.run_pipeline \
  --input path/to/song.mp3 \
  --out outputs/arrangement.mid
```

### Launching the Streamlit web app locally

```bash
streamlit run src/app/main.py
```

The app opens at http://localhost:8501.

## 6. Training

Training is intended for Google Colab (free T4 GPU). Open the notebook:

```
notebooks/03_train.ipynb
```

Run cells in order. Total wall-clock time on T4: ~2 hrs Phase A (JSB pretrain) + ~2 hrs Phase B (jaCappella fine-tune).

To train locally on CUDA:

```bash
python -m src.training.train --config configs/train.yaml
```

## 7. Reproducing evaluation

```bash
python -m src.eval.evaluate --checkpoint models/hybrid_finetuned.pt
python -m src.eval.evaluate --checkpoint models/transformer_baseline.pt
python -m src.eval.ablation --sweep voice_leading,architecture
```

Outputs land in `reports/` as Markdown summaries and PNG plots.

## 8. Deployment

The live app is hosted on Streamlit Community Cloud. To deploy your own fork:

1. Push this repo to GitHub.
2. Connect the repo at https://share.streamlit.io.
3. Set entrypoint: `src/app/main.py`.
4. Secrets (if any) go in the Streamlit Cloud dashboard, not in git.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `demucs` import error | Ensure `ffmpeg` is on PATH. |
| CUDA out of memory during training | Reduce `batch_size` in `configs/train.yaml`. |
| Streamlit app hangs on first upload | First inference downloads Demucs weights (~2 GB). Subsequent runs are fast. |
| `music21` can't find chorales | Run `python -c "from music21 import corpus; corpus.chorales.Iterator()"` once to populate the cache. |
| `crepe` TensorFlow version conflict | Pin `tensorflow>=2.12,<2.16` in your environment. |
