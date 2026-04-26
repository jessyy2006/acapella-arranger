# `data/` — Datasets

This directory is **gitignored** — large data files are never committed. After cloning, run the download script to populate it:

```bash
python scripts/download_data.py
```

That script materializes:

| Subdirectory | Contents | Source |
|---|---|---|
| `data/raw/jacappella/` | jaCappella corpus MusicXML scores (50 songs, 10 genres) | HuggingFace, gated dataset — set `HF_TOKEN` if needed |
| `data/raw/jsb_chorales/` | JSB Chorales (~278 clean SATB chorales after filtering) | `music21.corpus.chorales.Iterator()` |
| `data/processed/{train,val,test}_{jsb,jacappella}.pt` | Pre-tokenized + augmented `SATBDataset` objects, written by `scripts/prepare_data.py` | Generated locally |

Disk footprint: ~50 MB raw + ~10 MB processed.

The 70 / 15 / 15 train / val / test split is **by song**, with `seed=42`, executed in `src/data/loaders.py:build_split` so a single song never appears in two splits. See `scripts/prepare_data.py` for full preprocessing flags (window size, transposition range, per-source song limits).

To rebuild processed splits from scratch:

```bash
python scripts/prepare_data.py --force
```
