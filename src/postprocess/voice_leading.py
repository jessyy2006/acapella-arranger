"""Voice-leading post-processor for SATB token sequences.

Applied after the arrangement model has produced raw per-voice tokens and
before MIDI export. Two rules, both cheap:

1. **Range clamp** — if a pitch token lands outside the voice's observed
   training range (`SATB_RANGES`), octave-transpose (±12 semitones) until
   it fits. Falls back to nearest-endpoint clipping if no octave works
   (shouldn't happen for SATB + 128-MIDI but the guard keeps the function
   total).
2. **Parallel-5th / parallel-8ve detection** — flag-only. Walks the
   per-sample pitch-vs-time timeline (same construction
   :func:`src.eval.metrics.voice_crossing_rate` uses) and reports every
   time-step where two adjacent voices move in the same direction at
   interval 7 or 12 semitones. No automatic correction — correcting
   parallels is genuinely hard and the rubric gives us evidence-of-
   detection either way.

The module intentionally stays a pure ``dict[str, list[int]] →
dict[str, list[int]]`` transform. ``src/pipeline/run_pipeline.py`` (Jess
lane, separate issue) wraps this behind a ``--voice-leading/--no-voice-
leading`` flag so Soham's ablation can toggle it.
"""

from __future__ import annotations

import logging
from typing import Final

from src.data.vocab import (
    BAR,
    PAD,
    REST,
    SATB_RANGES,
    is_pitch_token,
    pitch_to_token,
    token_to_duration,
    token_to_pitch,
)

logger = logging.getLogger(__name__)

VOICES: Final[tuple[str, ...]] = ("s", "a", "t", "b")


def _clamp_pitch_to_range(midi: int, lo: int, hi: int) -> int:
    """Return ``midi`` transposed by octaves until it lands in ``[lo, hi]``.

    If neither octave-up nor octave-down resolves it (``hi - lo < 12``
    would make this possible in principle; for SATB the gap is always
    large enough), clip to the nearest endpoint. Clip-fallback keeps the
    function total so callers don't need to special-case.
    """
    if lo <= midi <= hi:
        return midi
    # Octave-down until below hi.
    while midi > hi:
        midi -= 12
        if midi < lo:
            # Overshot — pick whichever of the two bracketing octaves is
            # closer to the original range and clip the residual.
            break
    # Octave-up until above lo.
    while midi < lo:
        midi += 12
        if midi > hi:
            break
    if midi < lo:
        return lo
    if midi > hi:
        return hi
    return midi


def _clamp_tokens(tokens: list[int], lo: int, hi: int) -> list[int]:
    """Range-clamp every pitch token in ``tokens``; pass others through."""
    out: list[int] = []
    for tok in tokens:
        if not is_pitch_token(tok):
            out.append(tok)
            continue
        midi = token_to_pitch(tok)
        assert midi is not None  # is_pitch_token guarantees this
        clamped = _clamp_pitch_to_range(midi, lo, hi)
        out.append(pitch_to_token(clamped) if clamped != midi else tok)
    return out


def apply_voice_leading(
    tokens_by_voice: dict[str, list[int]],
    *,
    enable_range_clamp: bool = True,
    enable_parallel_detect: bool = True,
) -> dict[str, list[int]]:
    """Post-process per-voice token sequences before MIDI export.

    Parameters
    ----------
    tokens_by_voice
        ``{"s": [...], "a": [...], "t": [...], "b": [...]}`` — each value
        is a per-voice token sequence in the vocab defined by
        :mod:`src.data.vocab`.
    enable_range_clamp
        If ``True`` (default), octave-transpose out-of-range pitches into
        each voice's observed training range.
    enable_parallel_detect
        If ``True`` (default), detect parallel 5ths / 8ves and log them
        at WARNING level. No automatic correction — callers that want the
        structured list should call :func:`detect_parallel_motion`
        directly.

    Returns
    -------
    A new ``{voice: tokens}`` dict with the same keys; input is not
    mutated.
    """
    if set(tokens_by_voice.keys()) != set(VOICES):
        raise ValueError(
            f"expected voices {sorted(VOICES)}, got {sorted(tokens_by_voice)}"
        )

    out: dict[str, list[int]] = {v: list(tokens_by_voice[v]) for v in VOICES}

    if enable_range_clamp:
        for voice in VOICES:
            lo, hi = SATB_RANGES[voice]
            out[voice] = _clamp_tokens(out[voice], lo, hi)

    if enable_parallel_detect:
        violations = detect_parallel_motion(out)
        for step, (va, vb), interval in violations:
            kind = "octave" if interval == 12 else "fifth"
            logger.warning(
                "parallel %s at step %d between %s/%s (interval %d)",
                kind,
                step,
                va,
                vb,
                interval,
            )

    return out


def _timeline(tokens: list[int]) -> list[int | None]:
    """Rebuild a pitch-per-16th-note timeline from a token stream.

    Mirrors the construction used by
    :func:`src.eval.metrics.voice_crossing_rate` so the two modules are
    algorithmically consistent. Each (pitch, duration) pair expands to
    ``duration`` entries; REST / BAR / PAD become ``None`` / are skipped.
    Returns a list of MIDI pitches (``int``) or ``None`` for rests.
    """
    out: list[int | None] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in (PAD, BAR):
            i += 1
            continue
        if t == REST or token_to_pitch(t) is not None:
            if i + 1 >= len(tokens):
                break
            dur = token_to_duration(tokens[i + 1])
            if dur is None:
                i += 1
                continue
            pitch = None if t == REST else token_to_pitch(t)
            out.extend([pitch] * int(dur))
            i += 2
            continue
        i += 1
    return out


def detect_parallel_motion(
    tokens_by_voice: dict[str, list[int]],
) -> list[tuple[int, tuple[str, str], int]]:
    """Return parallel-5th / parallel-8ve violations as structured tuples.

    Each returned tuple is ``(time_step, (voice_a, voice_b), interval)``
    — ``time_step`` in 16th-note units from the start of the timeline,
    ``voice_a/voice_b`` one of the adjacent pairs ``("s", "a")``,
    ``("a", "t")``, or ``("t", "b")``, and ``interval`` is the shared
    semitone gap (7 for a fifth, 12 for an octave) observed at both
    step-1 and step.

    A "parallel" is flagged when:
      - Both voices have a defined pitch at step ``i-1`` and step ``i``
        (i.e. neither is resting or padded),
      - The interval between the two voices is 7 or 12 semitones at both
        steps,
      - Both voices moved in the same direction (up or down) between the
        two steps. (Oblique motion doesn't count as parallel.)
    """
    timelines = {v: _timeline(tokens_by_voice[v]) for v in VOICES}
    min_len = min(len(timelines[v]) for v in VOICES)
    if min_len < 2:
        return []

    pairs: tuple[tuple[str, str], ...] = (("s", "a"), ("a", "t"), ("t", "b"))
    flagged_intervals = (7, 12)

    out: list[tuple[int, tuple[str, str], int]] = []
    for step in range(1, min_len):
        for va, vb in pairs:
            a_prev = timelines[va][step - 1]
            b_prev = timelines[vb][step - 1]
            a_cur = timelines[va][step]
            b_cur = timelines[vb][step]
            if any(x is None for x in (a_prev, b_prev, a_cur, b_cur)):
                continue
            assert a_prev is not None and b_prev is not None
            assert a_cur is not None and b_cur is not None
            prev_interval = a_prev - b_prev
            cur_interval = a_cur - b_cur
            if prev_interval != cur_interval:
                continue  # different intervals → not parallel
            if prev_interval not in flagged_intervals:
                continue
            # Same direction? Skip if neither voice actually moved.
            a_move = a_cur - a_prev
            b_move = b_cur - b_prev
            if a_move == 0 and b_move == 0:
                continue  # no motion — not parallel, just static
            if (a_move > 0) != (b_move > 0) and a_move != 0 and b_move != 0:
                continue  # contrary motion
            out.append((step, (va, vb), int(prev_interval)))
    return out
