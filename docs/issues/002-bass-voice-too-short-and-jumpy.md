# 002 — Bass voice generates too-short, jumpy notes

- **Type**: bug (model-output quality)
- **Priority**: normal — doesn't block, but bass is a foundational voice so quality hit is audible across the whole arrangement
- **Effort**: small (first fix); medium if per-voice postprocessing is also needed
- **Found in**: Phase 4 smoke arrangement on `outputs/smoke_input.mp3`
- **Commit at time of report**: `0889918`

## TL;DR

Generated bass (B) is a rapid, jumpy sequence of short notes. Pop bass should be mostly long held notes (half / whole / dotted-half) with occasional quarter notes and very few short notes. The bass voice shouldn't be pitching around every beat.

## Current behaviour

Every smoke run on `outputs/smoke_arrangement.mid` since the pipeline landed:

- Bass consistently hits `max_len = 256` tokens (implies short durations packing many notes into the budget).
- Pitch changes every few notes.
- Listening: dissonant and restless, doesn't "ground" the chord progression.

## Expected behaviour

For pop-style material:

- Default note length: half / whole / dotted-half.
- Quarter notes as secondary rhythmic level.
- 8ths occasional, 16ths rare.
- Pitch changes roughly once per chord, not once per beat.

## Candidate root causes

1. **Uniform sampling config across the four voices.** `src/pipeline/run_pipeline.run_pipeline` calls `generate_voice_tokens` with `duration_temperature=1.1` for all of S / A / T / B. That temperature was tuned on listening tests to keep the upper voices from collapsing onto a single quarter-note bucket — the exact *opposite* of what bass needs. Same issue in `scripts/sample_midi.py`.
2. **Training distribution.** Model was trained on JSB chorales + jacappella. JSB bass is active (baroque walking-line style); jacappella is a cappella pop which may or may not match. Neither corpus is pop-band bass where the genre convention is "root on 1, hold it."
3. **No voice-specific postprocessing.** `src/postprocess/voice_leading.apply_voice_leading` handles range-clamp + parallel-motion detection but doesn't touch rhythm.

## Fix plan (proposed order)

1. **Per-voice sampling config.** Expose per-voice overrides on `run_pipeline`: keep `duration_temperature=1.1` for S/A/T, override bass to **`duration_temperature=0.6-0.8`**. Lower temperature concentrates duration sampling on the modal bucket (quarter or longer), which is exactly what bass needs. Also consider dropping bass `temperature` from 0.5 to 0.3 for more stable pitch choices. Tiny API change in `run_pipeline` and `scripts/sample_midi.py`.
2. **If (1) insufficient: bass-specific coalesce pass.** Add a postprocess in `src/postprocess/voice_leading.py` that (a) merges consecutive same-pitch bass notes, and (b) absorbs any bass note shorter than `min_bass_sixteenths` (default 4 = quarter note) into its neighbour. Mirrors the audio-pipeline `_merge_short_runs` pattern.
3. **If (1) + (2) insufficient: retrain or condition differently.** Out of scope here — would require pop bass training data or a genre-conditioning head. Only pursue if (1) + (2) left the bass still wrong.

## Acceptance criteria

- Listening in MuseScore: bass sounds like a pop bass line — mostly sustained notes, pitch changing roughly at chord boundaries rather than every beat.
- Bass token count drops substantially from 256 (hitting max_len) — expect ~80-120 tokens for a 20-second clip if durations are genuinely longer.
- No regression in S / A / T quality on the same smoke clip.

## Relevant files

- `src/pipeline/run_pipeline.py` — the per-voice loop around `generate_voice_tokens`
- `scripts/sample_midi.py` — same loop pattern, needs same treatment
- `src/pipeline/decode.py` — `generate_voice_tokens` already accepts the kwargs; no signature change needed
- `src/postprocess/voice_leading.py` — home for the fix-plan-step-2 coalesce pass if we need it

## Risks / notes

- Lowering bass temperatures sacrifices rhythmic interest by construction. Per the user's stated aesthetic ("long, held out notes that DO NOT CHANGE very often"), this is the intended tradeoff.
- Per-voice sampling is a small API expansion. Default behaviour unchanged if no overrides are provided.
- Bass is the voice most likely to benefit from this pattern, but the same override mechanism could later apply to T if tenor also ends up too busy — build the API with all four voices in mind.
