# 003 — SATB voices should follow section-level patterns, not sound random throughout

- **Type**: feature (structural / aesthetic)
- **Priority**: normal — doesn't block delivery but is the biggest remaining audible-quality gap in the arrangement
- **Effort**: **large** (first attempt is medium; full fix likely requires model-side changes)
- **Found in**: smoke arrangement on `outputs/smoke_input.mp3`
- **Commit at time of report**: `39ca3d0`
- **Depends on**: issue 001 (Lead must be sensible for the model to condition on), issue 002 (bass needs the coalesce fix to stop drowning out structure)

## TL;DR

Real arrangements reuse motifs across matching sections of a song — verse 1 and verse 2 share the same bass/alto/tenor figure, chorus 1 and chorus 2 share a different one, and a bridge is typically distinctive. Our current pipeline generates each voice autoregressively with no section awareness, so the SATB lines sound random throughout rather than grounded in a structure. Jess wants "pattern A for a couple of measures, then no pattern for some measures, then pattern B for some measures" — motivic repetition tied to the song's form.

## Current behaviour

- `src/pipeline/run_pipeline.run_pipeline` calls `generate_voice_tokens` once per voice on the full lead. The model has no "where in the song am I" signal.
- Each voice decoder attends to the encoded lead memory but doesn't know which lead-tokens it's re-visiting.
- Listener hears a moment-to-moment plausible chorale that never settles into a repeated idea.

## Expected behaviour

- Verse 1 and verse 2 (same lead contour) should get the **same** SATB motif, not independently-resampled variants.
- Distinct sections (verse / chorus / bridge) should get distinct but consistent motifs.
- Free passages (fills, ad-libs) can keep the current free-generation behaviour.

## Candidate root causes

1. **No section detection on the lead.** We don't know where the song's sections are, so we can't even attempt to reuse generated content.
2. **No mechanism to repeat a generated segment.** Even with section boundaries, `generate_voice_tokens` always samples fresh — there's no "replay these tokens at this position" primitive.
3. **Model wasn't trained with section conditioning.** The hybrid decoder has no section-id input; retraining with that signal is out of scope for the remaining time.

## Fix plan (proposed order)

1. **Lead-side section detection + copy-paste generation (medium).** First real attempt:
   - Add `src/pipeline/sections.py`: use `librosa.segment` (beat-sync self-similarity + agglomerative clustering) on the isolated vocal or original audio to detect section boundaries and label sections by similarity.
   - In `run_pipeline`, for each unique section label, generate SATB once on just that section's lead tokens. At positions where that label repeats, paste the generated SATB tokens (with voice-leading cleanup at seams).
   - Deterministic seeding per section label so verse 1 and verse 2 get bitwise-identical SATB if their leads are musically identical.
   - This alone should produce the pattern-A-then-pattern-B effect Jess asked for.
2. **Cross-section seam smoothing (small, needed after 1).** Where two different sections meet, the pasted SATB will have voice-leading jumps at the boundary. Extend `src/postprocess/voice_leading.py` with a "seam smoothing" pass that octave-transposes or nudges pitches across the first 1-2 notes of the new section for continuity.
3. **If (1) + (2) insufficient: motif-level repetition within a section (medium/large).** Even inside one section, pop vocals often have phrase-level motifs (every 2 bars). Detect sub-section motifs in the lead via token-level self-similarity; generate the first motif and replay.
4. **If (1)-(3) insufficient: retrain with section conditioning (large, out of scope).** Add a section-label embedding to the lead encoder. Would need a dataset with section annotations — jacappella might have them; JSB mostly won't. Defer.

## Acceptance criteria

- Repeated sections of the lead produce bitwise-identical (or near-identical) SATB output.
- Section boundaries audible in listening — a clean "new idea starts here" at verse/chorus joins rather than a continuous stream of unrelated material.
- Regression check: on a single-section lead (like `outputs/smoke_input.mp3` if no sections are detected), the behaviour degrades gracefully to the current pipeline — no new bugs where the old path worked.

## Relevant files

- `src/pipeline/run_pipeline.py` — the per-voice generation loop (lines ~115-135). Needs to split by section and aggregate.
- `src/pipeline/decode.py` — `generate_voice_tokens` accepts a `seed` kwarg already; plumb a per-section seed through from above.
- `src/pipeline/sections.py` — **new file**. Section detection from the audio (or tokens) + label assignment.
- `src/postprocess/voice_leading.py` — seam smoothing for step 2.
- Tests: new `tests/test_sections.py` for segmentation; augment `tests/test_run_pipeline.py` for the multi-section path.

## Risks / notes

- **Section detection is noisy.** `librosa.segment` is solid but not perfect on pop; expect ~10-20% boundary disagreement with a human labeller. The fallback to single-section gen should be clean so bad detection doesn't make the pipeline worse.
- **"Bitwise-identical SATB across repeats" may sound too mechanical.** Real performers vary motifs subtly each repeat. Consider a small perturbation (e.g., a fresh seed every N repeats, or a low-temperature resample) if the listener complains about rigidity. Orthogonal tuning; ship the copy-paste baseline first.
- **Seam smoothing interacts with the bass coalesce from issue 002.** Bass is already heavily smoothed; pasting identical bass across sections + then coalescing could produce very static output. Consider running coalesce *before* the section paste, or per-section, to avoid over-smoothing.
- **Not a rubric line item, but quality-perception-dominant.** The grader's "this sounds like a real arrangement" vs "this sounds like a language model" gut-check will almost certainly hinge on whether sections recur audibly. High leverage for perceived quality even though the rubric can't point at a specific test.
