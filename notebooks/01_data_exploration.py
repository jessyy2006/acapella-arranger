# %% [markdown]
# # Data Exploration — jaCappella + JSB Chorales
#
# Goal: understand both datasets well enough to design a unified tokenizer
# and DataLoader. Run top-to-bottom after `scripts/download_data.py` completes.
#
# Can be run as a plain script, or opened in VS Code's interactive mode,
# or converted to a notebook via `jupytext --to notebook 01_data_exploration.py`.
#
# Answer these by the end:
# 1. Exact part names in jaCappella MusicXML (expected: Lead Vocal,
#    Soprano, Alto, Tenor, Bass, Vocal Percussion).
# 2. Whether JSB part names are consistently Soprano / Alto / Tenor / Bass.
# 3. Global MIDI range across both datasets → vocabulary size.
# 4. Distribution of note durations → quantization grid choice (16th? 32nd?).
# 5. Whether jaCappella "Lead Vocal" ≈ Soprano or is genuinely distinct.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from music21 import converter, corpus, note

# Resolve project root whether running from notebooks/ or repo root.
_here = Path.cwd()
PROJECT_ROOT = _here.parent if _here.name == "notebooks" else _here
JACAPPELLA_DIR = PROJECT_ROOT / "data" / "raw" / "jacappella"

print(f"Project root: {PROJECT_ROOT}")
print(f"jaCappella dir: {JACAPPELLA_DIR}  (exists: {JACAPPELLA_DIR.exists()})")

# %% [markdown]
# ## 1. jaCappella — metadata overview

# %%
meta_path = JACAPPELLA_DIR / "meta.csv"
if meta_path.exists():
    meta = pd.read_csv(meta_path)
    print(f"Rows: {len(meta)}")
    print(f"Columns: {list(meta.columns)}")
    print()
    print(meta.head())
else:
    print("meta.csv not found — run scripts/download_data.py first.")
    meta = None

# %%
if meta is not None and "subset" in meta.columns and "title_in_en" in meta.columns:
    per_subset = meta.groupby("subset")["title_in_en"].nunique().sort_values(ascending=False)
    print("Songs per subset:")
    print(per_subset)
    print(f"\nUnique songs total: {meta['title_in_en'].nunique()}")

# %%
if meta is not None and "voice_part" in meta.columns:
    print("Voice-part row counts in meta.csv:")
    print(meta["voice_part"].value_counts())

# %% [markdown]
# ## 2. jaCappella — inspect one MusicXML score

# %%
all_xml = sorted(JACAPPELLA_DIR.rglob("*.musicxml"))
# Filter out the SVS / romaji lyric variants; keep the main scores.
main_xml = [
    p for p in all_xml
    if not p.name.endswith("_SVS.musicxml")
    and not p.name.endswith("_romaji.musicxml")
]
print(f"Total .musicxml files: {len(all_xml)}")
print(f"Main scores (excluding SVS / romaji variants): {len(main_xml)}")

# %%
if main_xml:
    sample_path = main_xml[0]
    print(f"Inspecting: {sample_path.relative_to(JACAPPELLA_DIR)}")
    sample_score = converter.parse(str(sample_path))
    print(f"Parts: {len(sample_score.parts)}")
    for p in sample_score.parts:
        n_notes = len(list(p.flatten().notes))
        print(f"  - {p.partName or p.id}: {n_notes} notes, {float(p.highestTime):.1f} qL")

# %% [markdown]
# ## 3. jaCappella — per-voice pitch range aggregated across all songs

# %%
jacap_voice_stats: dict[str, list[tuple[int, int, int]]] = {}
jacap_song_lengths: list[float] = []
jacap_failed = 0

for xml_path in main_xml:
    try:
        sc = converter.parse(str(xml_path))
    except Exception as e:
        print(f"  skip {xml_path.name}: {e}")
        jacap_failed += 1
        continue

    jacap_song_lengths.append(float(sc.highestTime))

    for part in sc.parts:
        name = (part.partName or part.id or "unknown").strip()
        notes = [n for n in part.flatten().notes if isinstance(n, note.Note)]
        if not notes:
            continue
        pitches = [n.pitch.midi for n in notes]
        jacap_voice_stats.setdefault(name, []).append(
            (min(pitches), max(pitches), len(pitches))
        )

print(f"Parsed: {len(main_xml) - jacap_failed} / {len(main_xml)}")
print()
print(f"{'Voice':22s}  min-MIDI-range   max-MIDI-range   notes (median)   songs")
for voice, records in sorted(jacap_voice_stats.items()):
    mins = [r[0] for r in records]
    maxs = [r[1] for r in records]
    ns = sorted(r[2] for r in records)
    median_notes = ns[len(ns) // 2]
    print(
        f"{voice:22s}  "
        f"[{min(mins):3d}, {max(mins):3d}]        "
        f"[{min(maxs):3d}, {max(maxs):3d}]        "
        f"{median_notes:4d}             "
        f"{len(records)}"
    )

# %%
if jacap_song_lengths:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(jacap_song_lengths, bins=20)
    ax.set_xlabel("Song length (quarter-length)")
    ax.set_ylabel("Count")
    ax.set_title("jaCappella — song duration distribution")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. JSB Chorales — via music21

# %%
chorales = list(corpus.chorales.Iterator())
print(f"JSB chorales available: {len(chorales)}")

if chorales:
    first = chorales[0]
    title = first.metadata.title if first.metadata else "(untitled)"
    print(f"\nFirst chorale: {title}")
    print(f"Parts: {len(first.parts)}")
    for p in first.parts:
        n_notes = len(list(p.flatten().notes))
        print(f"  - {p.partName}: {n_notes} notes, {float(p.highestTime):.1f} qL")

# %% [markdown]
# ## 5. JSB — per-voice pitch range across all chorales

# %%
jsb_voice_stats: dict[str, list[tuple[int, int, int]]] = {}
jsb_lengths: list[float] = []
jsb_failed = 0

for ch in chorales:
    try:
        jsb_lengths.append(float(ch.highestTime))
        for part in ch.parts:
            name = (part.partName or "unknown").strip()
            notes = [n for n in part.flatten().notes if isinstance(n, note.Note)]
            if not notes:
                continue
            pitches = [n.pitch.midi for n in notes]
            jsb_voice_stats.setdefault(name, []).append(
                (min(pitches), max(pitches), len(pitches))
            )
    except Exception:
        jsb_failed += 1

print(f"Processed: {len(chorales) - jsb_failed}  Failed: {jsb_failed}")
print()
print(f"{'Voice':22s}  min-MIDI-range   max-MIDI-range   notes (median)   pieces")
for voice, records in sorted(jsb_voice_stats.items()):
    mins = [r[0] for r in records]
    maxs = [r[1] for r in records]
    ns = sorted(r[2] for r in records)
    median_notes = ns[len(ns) // 2]
    print(
        f"{voice:22s}  "
        f"[{min(mins):3d}, {max(mins):3d}]        "
        f"[{min(maxs):3d}, {max(maxs):3d}]        "
        f"{median_notes:4d}             "
        f"{len(records)}"
    )

# %%
if jsb_lengths:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(jsb_lengths, bins=20)
    ax.set_xlabel("Chorale length (quarter-length)")
    ax.set_ylabel("Count")
    ax.set_title("JSB Chorales — duration distribution")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Note duration histogram — informs quantization grid

# %%
def collect_durations(score_iter, limit=None):
    durs = []
    for i, sc in enumerate(score_iter):
        if limit and i >= limit:
            break
        for part in sc.parts:
            for n in part.flatten().notes:
                durs.append(float(n.quarterLength))
    return durs

# Sample first 50 chorales + 10 jaCappella songs to keep this fast.
jsb_durations = collect_durations(chorales[:50])

jacap_sample = []
for p in main_xml[:10]:
    try:
        jacap_sample.append(converter.parse(str(p)))
    except Exception:
        pass
jacap_durations = collect_durations(jacap_sample)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(jsb_durations, bins=30, range=(0, 4))
axes[0].set_title(f"JSB — note durations (qL), n={len(jsb_durations)}")
axes[0].set_xlabel("quarter-length")
axes[1].hist(jacap_durations, bins=30, range=(0, 4))
axes[1].set_title(f"jaCappella — note durations (qL), n={len(jacap_durations)}")
axes[1].set_xlabel("quarter-length")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Global pitch vocabulary across both datasets

# %%
def flatten_pitches(voice_stats):
    lo = min(r[0] for records in voice_stats.values() for r in records)
    hi = max(r[1] for records in voice_stats.values() for r in records)
    return lo, hi

if jacap_voice_stats:
    ja_lo, ja_hi = flatten_pitches(jacap_voice_stats)
    print(f"jaCappella global MIDI range: [{ja_lo}, {ja_hi}]  span = {ja_hi - ja_lo + 1}")
if jsb_voice_stats:
    jsb_lo, jsb_hi = flatten_pitches(jsb_voice_stats)
    print(f"JSB global MIDI range:        [{jsb_lo}, {jsb_hi}]  span = {jsb_hi - jsb_lo + 1}")

if jacap_voice_stats and jsb_voice_stats:
    g_lo = min(ja_lo, jsb_lo)
    g_hi = max(ja_hi, jsb_hi)
    print(f"Combined global MIDI range:   [{g_lo}, {g_hi}]  span = {g_hi - g_lo + 1}")
    print("(Vocabulary should cover this range plus special tokens: PAD, SOS, EOS, REST.)")
