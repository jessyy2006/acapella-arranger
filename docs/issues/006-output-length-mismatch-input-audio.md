# 006 — Output MIDI length doesn't match the input MP3 duration

- **Type**: bug
- **Priority**: high (user-facing artefact quality — downstream video sync requires exact duration)
- **Effort**: medium (required four distinct fixes, not one)
- **Found in**: user-requested check on `outputs/smoke_input.mp3` after issue 005 was parked
- **Commit at time of report**: `d73e51e`
- **Status**: fixed in `c375fb7`

## TL;DR

On a 62.414 s input MP3 the pipeline was producing a ~64 s output MIDI — and more importantly the five parts (Lead + SATB) each ended at a different sixteenth count. Nothing in the pipeline enforced "output total = input total". Fix required four separate changes because drift was coming from four independent layers: audio-to-tokens quantisation, per-voice stochastic generation, music21's MIDI writer, and pretty_midi's tempo rounding.

## Current behaviour (pre-fix)

Smoke run on `outputs/smoke_input.mp3` (62.414 s):

- Lead ended at ~59 s (quantisation-dependent; varied).
- S / A / T / B each ended at different lengths governed by stochastic EOS + per-voice coalesce + grid-align.
- Exported MIDI was padded by music21 to ~64 s (one bar of 4/4 rounding on the final Measure).
- No tempo marker in the output, so playback defaulted to 120 BPM and skewed any tempo ≠ 120.

## Expected behaviour

- `mido.MidiFile(out_path).length == input_mp3_seconds` within one MIDI tick (≤ 1 ms on a 60 s clip).
- Every track (meta + Lead + S/A/T/B) ends at the exact same tick.
- The file opens in GarageBand / Logic / MuseScore without a "download additional sounds" prompt.

## Root causes — four-layer drift

1. **Per-voice token-count drift.** `fit_to_length` did not exist; `run_pipeline` concatenated whatever tokens the model emitted (`decode.py:80–120` stops at predicted EOS or `max_len=256`, not at a target duration) and passed them through voice-leading (`voice_leading.py:251–274`) which merges, drops, and grid-snaps events — none of which preserves total duration.

2. **music21 Measure padding.** `score.write("midi")` pads the final `stream.Measure` to a full 4/4 bar using `barDuration`, so a 27-full-measures + 7-sixteenths-partial Part gets exported as 28 full measures (= 448 sixteenths vs. target 439). Verified end-to-end by building a Part whose internal `duration.quarterLength` was 109.75 and watching the written MIDI round-trip back as 112.0.

3. **pretty_midi tempo-microseconds rounding.** First swap was to `pretty_midi` to avoid (2). That moved the drift from +9 sixteenths to +0.13 sixteenths — smaller but still 18 ms over on the smoke clip. Cause: pretty_midi's `PrettyMIDI(initial_tempo=X)` builds an internal tick scale whose seconds-per-tick is `60 / (X * resolution)`, and when the file is written this is converted to an integer `set_tempo` microseconds-per-quarter. Round-trip tempo for `initial_tempo=105.5` in a single-instrument test came back at 105.500070 — but the real pipeline's five-instrument export came back at 105.468915. Source of the inconsistency not located; switched to writing with `mido` directly.

4. **Trailing-REST padding invisible in MIDI.** `fit_to_length` pads short voices by appending a REST event. REST advances `t_sec` but emits no `note_on/note_off`. Without an explicit `end_of_track` at the target tick, a padded voice's track ended at its last note — soprano finished at 59.4 s while the other tracks reached 62.4 s.

## Fix applied (`c375fb7`)

1. New `fit_to_length(tokens, target_sixteenths)` in `src/postprocess/voice_leading.py`. Trims the tail (shortening the last event, dropping only when needed) or pads with a single trailing REST. Re-emits via the existing `_events_to_tokens` decomposer, which already handles any positive integer sixteenth count because `1 ∈ DUR_BUCKETS`.

2. New `_write_exact_midi` in `src/pipeline/run_pipeline.py` using `mido` at 960 PPQ (divisible by every entry of `DUR_BUCKETS`). Writes one conductor track with `set_tempo` + `end_of_track` at the target tick, then one track per voice with explicit `program_change=0` (GM Acoustic Grand Piano — fixes GarageBand's DLS-missing prompt). Each track gets an `end_of_track` meta at exactly `target_sixteenths * 240` ticks, so every track runs the full duration.

3. MIDI tempo is computed as `midi_tempo = 60 * target_sixteenths / (4 * target_duration_sec)` rather than using the tokenizer's estimated tempo. Differs from the tokenizer tempo by well under 0.1 BPM but makes `target_sixteenths * sec_per_16(midi_tempo) == target_duration_sec` exactly.

4. `_assemble_score` retained but the pipeline no longer uses its output — kept because `tests/test_run_pipeline.py` asserts on the in-memory part structure. The `tempo_bpm` kwarg was made optional so that test doesn't need rewriting.

## Acceptance criteria (verified)

- `ffprobe` on `outputs/smoke_input.mp3` → 62.413787 s.
- `mido.MidiFile('outputs/smoke_arrangement.mid').length` → 62.413837 s.
- Delta: 0.050 ms (one MIDI tick is ~0.59 ms at this tempo; we are sub-tick).
- All six tracks (conductor + Lead + S + A + T + B) end at tick 105360.
- Every track has `program_change=0`.
- File opens in GarageBand without prompting for downloads.
- Full suite 189/189; 8 new tests for `fit_to_length`.

## Known limitations (not addressed; candidates for future tickets)

- **Per-section alignment within the song is still off.** Section-aware generation (issue 003) caches a generated block per unique label and pastes it at every occurrence. Each paste's length is whatever the cached block happens to be, not the true wall-clock window of that occurrence. So if verse-2 starts at 45 s in the MP3, the output MIDI's verse-2 doesn't land there — only the overall end matches. Fixing this would require re-fitting the pasted block to each occurrence's actual duration, which breaks the "verse 1 tokens == verse 2 tokens" invariant from issue 003.
- **Trim can shorten the final chord** when a voice overshoots. Musically benign but the penultimate chord can feel clipped by a 16th or two; the alternative (dropping the chord entirely) is worse.

## Relevant files

- `src/postprocess/voice_leading.py:251–285` — `fit_to_length` + its event helpers.
- `src/pipeline/run_pipeline.py:100–195` — `_tokens_to_note_events`, `_write_exact_midi`, and the `midi_tempo` derivation.
- `src/pipeline/run_pipeline.py:218–295` — wire-in of `fit_to_length` + the new writer into `run_pipeline`.
- `tests/test_voice_leading.py` — `TestFitToLength` (8 cases: exact / pad / trim / trim-last-event / no-framing / empty / zero-target / negative-target).

## Risks / notes

- Writing the MIDI directly with `mido` means we've dropped music21's MIDI export path entirely from the production flow. Anything that used to rely on music21 setting default metadata (instrument names, key signatures) is now absent from the output file. That's intentional — we control every meta-event explicitly — but any downstream consumer that expected music21-flavoured metadata needs to be re-checked.
- Piano-for-every-track is a demo-quality choice, not a musical one. Future work could expose `--program-lead` / `--program-satb` flags, but only after confirming the target DAW has the relevant GM patches.
