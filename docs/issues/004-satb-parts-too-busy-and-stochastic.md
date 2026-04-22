# 004 — SATB parts too busy and stochastic; arrangement is messy

- **Type**: bug (output-quality / aesthetic)
- **Priority**: **high** — Jess flagged this directly; it's the dominant audible problem right now, bigger than any of the earlier issues when the whole arrangement is played together.
- **Effort**: medium
- **Found in**: smoke arrangement on `outputs/smoke_input.mp3` after issue 003's section-aware fix landed
- **Commit at time of report**: `da07a52`
- **Related**: issue 002 already addressed this for bass; this ticket extends the same treatment to S/A/T and tightens further. Issue 003's section-aware path is orthogonal (it controls repetition across sections) and should be kept.

## TL;DR

Soprano, alto, and tenor generate far too many short notes chained together, giving a messy, stochastic, "language-model" sound instead of defined rhythms and patterns. Each voice should be simpler and cleaner — repeating rhythmic figures, fewer notes per measure, less frantic 16th-note motion.

## Current behaviour

Most recent smoke (`da07a52`, section-aware on, bass coalesced):

- Per-voice note counts over the full 62 s clip:
  - S: 518
  - A: 523
  - T: 684 (busiest — ~11 notes per second)
  - B: 241 (already coalesced per issue 002; this is fine)
- Upper voices still run at `temperature=0.5`, `duration_temperature=1.1`. Those defaults were tuned against JSB chorales in listening tests to *avoid* quarter-note monotony; the consequence is the current restless output on pop material.
- No S/A/T equivalent of the bass coalesce pass landed in issue 002 — their token streams go through `apply_voice_leading`'s range-clamp and parallel-detect but are otherwise raw.

## Expected behaviour

For pop-style accompaniment:

- Upper voices should sustain block chords or simple rhythmic figures, not spit out a new note every sixteenth.
- Rhythms should be legible — a small vocabulary of figures (e.g. half + half, quarter + quarter + half, dotted-quarter + 8th + half) that recur rather than a fresh random pattern every bar.
- Note count per voice on a 62 s clip in a similar mood should be closer to **100-250** than 500-700.
- Overall: when you play lead + S + A + T + B together, it should sound like a chorale, not a crowd.

## Candidate root causes

1. **Temperature too hot for pop.** S/A/T defaults (`0.5 / 1.1`) were set against chorale / a cappella corpora where motivic activity is expected; on pop material we'd want lower temperatures across the board.
2. **No coalesce on upper voices.** The model emits short notes that the listener hears as frantic; the bass coalesce solves the same problem but only for bass. Generalise it to S/A/T with a smaller `min_sixteenths` floor so upper voices can still breathe.
3. **Section-aware generation compounds the problem.** Pre-issue-003 each voice was capped at `max_len=256` tokens across the whole song — naturally terse. Now we generate once per unique label and stitch, so each voice's total output is `num_labels * max_len` = up to `~512` tokens per voice. The upper bound on activity grew; defaults that were tuned against the old regime are now over-generous.

## Fix plan (proposed order)

1. **Per-voice sampling overrides for S/A/T (small).** Mirror the pattern added for bass in issue 002.
   - Expose `soprano_*`, `alto_*`, `tenor_*` temperature / duration-temperature kwargs on `run_pipeline` (and the `sample_midi` CLI).
   - Proposed defaults: `temperature=0.35`, `duration_temperature=0.8` for S/A/T (slightly hotter than bass's `0.3 / 0.7` — upper voices still need more variety than bass — but well below the current `0.5 / 1.1`).
   - Rationale: temperature alone can't move the learned mode, but it can sharpen it — fewer "off-mode" short notes get sampled.

2. **Upper-voice coalesce pass (small).** Extend the bass coalesce in `src/postprocess/voice_leading.py` to run on S/A/T too.
   - New kwarg: `upper_coalesce_min_sixteenths` (default 2 = 8th note).
   - Keep bass at its existing `min_sixteenths=3`. Upper voices at 2 lets them be slightly more active than bass but absorbs any 16th.
   - Generalise the existing `apply_voice_leading` block so it runs `coalesce_voice_tokens` per voice with a configurable floor; bass uses one setting, S/A/T use another.

3. **Listen + tune (user-in-the-loop).** After 1+2 ship, Jess listens. If S/A/T are too clean, raise their temperatures 0.05 at a time or drop the coalesce floor to 1 (no absorption). If still too busy, push the coalesce floor up to 3 for all four voices.

4. **If 1-3 insufficient: motivic repetition within a section (larger, out of scope).** Even inside one section, pop vocals have 2-4-bar repeating figures. Detect via token-level self-similarity and replay across the section. Overlap with issue 003 fix-plan step 3; worth attempting only if the temperature+coalesce combo isn't enough.

## Acceptance criteria

- Per-voice note counts on `outputs/smoke_input.mp3` drop to <300 for each of S/A/T, matching the "100-250 per voice" target above.
- Listening: when all five parts play together, the result sounds like a chorale or simple pop accompaniment, not a stochastic buzz. Individual voices should have audibly repeating rhythmic figures inside each section.
- No regression on bass (issue 002 criteria still satisfied): avg note length ≥ 1.0 quarter, pitch changes ≤ 1 per measure on average.

## Relevant files

- `src/pipeline/run_pipeline.py` — add `soprano_*` / `alto_*` / `tenor_*` kwargs to `run_pipeline`, dispatch them through the voice loop; CLI flags in `_parse_args`.
- `scripts/sample_midi.py` — same kwargs + CLI flags.
- `src/postprocess/voice_leading.py` — make the `coalesce_voice_tokens` call in `apply_voice_leading` run for all voices with per-voice `min_sixteenths`.
- `tests/test_voice_leading.py` — augment to cover the S/A/T coalesce case.
- `tests/test_run_pipeline.py` — add a test that per-voice kwargs round-trip to `generate_voice_tokens`.

## Risks / notes

- **Lowering S/A/T temperatures can flatten interesting moments.** The previous listening-test rationale for `1.1` duration-temperature still holds partly — too cold and every note becomes a quarter. Pick defaults cautiously; expose the knobs so Jess can re-tune after hearing.
- **Bass coalesce already compounds with section-aware paste.** Adding an upper-voice coalesce post-paste could over-flatten sections where S/A/T genuinely had phrase-level shape. If that happens, move the coalesce *inside* the per-section generation so each pasted section is coalesced locally. Orthogonal fix if needed.
- **High user-perception leverage.** Alongside issue 001 (lead) and 003 (sections), this is the last of the top-three "does this sound like a real arrangement?" levers. Fixing it should produce the biggest remaining subjective quality jump.
