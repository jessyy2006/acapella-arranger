# 005 — Tempo / time-signature / downbeat auto-detection for non-4/4 material

- **Type**: feature (quality improvement)
- **Priority**: **low** — parked for post-submission work; 4/4 assumption covers ~95% of pop material, and the audible problems we've debugged on `outputs/smoke_input.mp3` are generation-layer, not meter-detection.
- **Effort**: small (just tempo) → medium (tempo + time sig CLI flag) → large (full auto-detect with downbeat via madmom)
- **Found in**: exploration after issue 004 shipped. No active user complaint.
- **Commit at time of report**: `4c8bed7`

## TL;DR

The pipeline currently hardcodes 4/4 and starts measure 1 at t=0. Tempo is auto-detected via `librosa.beat.beat_track` but has no octave-error protection. For non-4/4 songs, pickup bars (anacruses), or songs where tempo estimation lands on the wrong multiple, measure alignment breaks — every bar drifts from there on. Feasible to fix, but not the highest-leverage quality work right now.

## Current behaviour

- **Tempo**: `audio_to_midi.py:657` — `librosa.beat.beat_track` on the vocal stem; falls back to 120 BPM.
- **Time signature**: hardcoded 4/4 in four places:
  - `audio_to_midi.py:604` — `meter.TimeSignature("4/4")` in the built Part.
  - `audio_to_midi.py:199` + `:601` — `sixteenth_sec = 60 / (4 * tempo_bpm)` assumes 4 sixteenths per beat.
  - `voice_leading.py:155` — `room = 16 - measure_pos` (one 4/4 bar = 16 sixteenths) in the coalesce re-emit.
  - `tokenizer.py` — 16th-note grid with no meter concept.
- **Downbeat offset**: not implemented. Quantization starts at t=0 regardless of where the song's first downbeat actually falls.

## Expected behaviour

Given any real pop / folk / gospel clip, the pipeline should land Lead + SATB bar lines where a human annotator would write them. Specifically:

1. Detect tempo (with octave-error guard) to within ±2 BPM of ground truth.
2. Identify time signature (at least 4/4 / 3/4 / 6/8 — the bulk of Western pop coverage).
3. Detect the first downbeat and offset quantization so measure 1 starts there.

## Candidate root causes

Not a bug — a scope limitation. The pipeline was written to the rubric spec and the smoke clip is 4/4. This ticket captures what's missing for broader real-world material.

## Fix plan (staged)

1. **Tempo hardening (small, 1 hr, no new deps).** Wrap `librosa.beat.beat_track` with an octave-error guard — run beat tracking with the detected BPM and with ±½ / ±2 multiples, pick the one with the highest onset-strength alignment. Log a warning when the choice is ambiguous.

2. **Manual time-signature CLI flag (small, 1-2 hrs, no new deps).** Add `--time-signature 4/4` / `--beats-per-bar 4` + `--beat-unit 4` through `run_pipeline` + `scripts/sample_midi.py`. Replace the 4/4 hardcodes with a threaded `(beats_per_bar, beat_unit)` pair: `frames_to_part`, `apply_voice_leading` → `_events_to_tokens`, and the `meter.TimeSignature` call. **This is the highest value / lowest cost step** — it lets us arrange 3/4 waltzes, 6/8 compound material, etc. by setting one flag when we know the meter.

3. **Auto time-signature detection (medium, 3-4 hrs, no new deps).** Heuristic based on onset-strength autocorrelation: find the period at which beat strength peaks (2, 3, or 4 beats per bar). Works reasonably on material with clear downbeats; can be gated behind a confidence threshold with manual-flag fallback. Risks misfires on syncopated pop.

4. **Full downbeat detection + tempo curve (large, 4-6 hrs, adds `madmom` dep).** Use `madmom.features.downbeats.RNNDownBeatProcessor + DBNDownBeatTrackingProcessor`. Gets beat times, downbeats, and time signature from one pretrained model. Cleanest output; caveat is madmom's install chain is fussy (pinned numpy / pytorch) and the package is ~20 MB.

5. **Time-varying tempo (very large, out of scope).** Support tempo changes across the song (rubato, accelerandi). Requires piecewise quantization and is non-trivial in music21. Defer until someone actually complains about it on a specific clip.

## Acceptance criteria

Depends on which step ships.

- Step 1 (tempo hardening): when the smoke clip's detected BPM is correct to within ±2 BPM, and on a test clip known to trip librosa's octave behaviour the pipeline now picks the right multiple.
- Step 2 (CLI flag): running `run_pipeline --time-signature 3/4` on a waltz produces a MIDI whose bar lines land at the intended positions.
- Step 3 (auto detect): the pipeline picks 3/4 on a waltz clip without the flag.
- Step 4 (downbeat): on a clip with a 2-beat pickup, measure 1 starts after the pickup, not at t=0.

## Relevant files

- `src/pipeline/audio_to_midi.py:157-210` (`pitch_track`), `:390-660` (`frames_to_part` + `extract_lead_tokens` tempo logic, 4/4 hardcodes).
- `src/postprocess/voice_leading.py:147-186` (`_events_to_tokens` — the `room = 16 - measure_pos` assumption).
- `src/pipeline/sections.py` — already detects section boundaries; the downbeat detector could share features (MFCC + onset strength) with it to keep the module count down.
- `src/data/tokenizer.py` — 16th-note grid is meter-agnostic; no changes needed unless we go beyond 16th resolution.

## Risks / notes

- **madmom vs. pure-librosa**: adding madmom is the "correct" engineering answer but inflates the install surface meaningfully. If this work ships pre-submission, keep pure-librosa. If post-submission, madmom is probably worth the cost.
- **Order matters**: tempo hardening first (free) → manual flag second (cheap, high-leverage) → auto-detect third (only if the flag proves annoying). Don't reach for downbeat detection unless actual pickup-bar misalignment is audible on a specific clip.
- **4/4 isn't universal but close**: the audible quality gap right now (issues 001-004) is ~100× bigger than what meter-detection would unlock. Don't redirect focus from generation to meter unless a non-4/4 clip lands on Jess's desk.
