"""Unified token vocabulary for SATB arrangement sequences.

Design
------
Each voice is serialised as an alternating stream of **pitch** and
**duration** tokens, with **BAR** tokens marking measure boundaries and
**SOS / EOS** framing the sequence. Rests use the dedicated REST token in
place of a pitch (followed by a duration token, like any other note).

Layout
------
    0            PAD     padding token
    1            SOS     start of sequence
    2            EOS     end of sequence
    3            BAR     measure boundary
    4            REST    rest (used in the pitch slot of a pair)
    5..132       PITCH   MIDI 0..127  (offset PITCH_OFFSET = 5)
    133..140     DUR     16th-grid buckets (offset DUR_OFFSET = 133)

Duration buckets are expressed in 16th-note units — the finest grid we
quantise to:

    1 = 16th    2 = 8th     3 = dotted 8th   4 = quarter
    6 = dotted  8 = half    12 = dotted half 16 = whole
        quarter

Total vocab size: 141.
"""

from __future__ import annotations

from typing import Final

PAD: Final[int] = 0
SOS: Final[int] = 1
EOS: Final[int] = 2
BAR: Final[int] = 3
REST: Final[int] = 4

PITCH_OFFSET: Final[int] = 5
NUM_PITCHES: Final[int] = 128  # MIDI 0..127

DUR_OFFSET: Final[int] = PITCH_OFFSET + NUM_PITCHES  # 133
DUR_BUCKETS: Final[tuple[int, ...]] = (1, 2, 3, 4, 6, 8, 12, 16)

VOCAB_SIZE: Final[int] = DUR_OFFSET + len(DUR_BUCKETS)  # 141

SPECIAL_TOKENS: Final[frozenset[int]] = frozenset({PAD, SOS, EOS, BAR, REST})


def pitch_to_token(midi: int) -> int:
    """Map a MIDI pitch (0..127) to its token id."""
    if not 0 <= midi < NUM_PITCHES:
        raise ValueError(f"MIDI pitch out of range: {midi}")
    return PITCH_OFFSET + midi


def token_to_pitch(token: int) -> int | None:
    """Inverse of ``pitch_to_token``. Returns None if token is not a pitch."""
    if PITCH_OFFSET <= token < DUR_OFFSET:
        return token - PITCH_OFFSET
    return None


def duration_to_token(sixteenths: float) -> int:
    """Quantise a duration (in 16th-note units) to the nearest bucket token.

    Durations shorter than the smallest bucket snap up to a 16th; longer than
    the largest snap down to a whole note. Ties between buckets snap up.
    """
    if sixteenths <= 0:
        raise ValueError(f"Duration must be positive, got {sixteenths}")
    # Pick the bucket minimising absolute distance; ties go to the larger
    # value (stable by design: sorted buckets, ``<=`` comparison).
    best = DUR_BUCKETS[0]
    best_diff = abs(sixteenths - best)
    for bucket in DUR_BUCKETS[1:]:
        diff = abs(sixteenths - bucket)
        if diff <= best_diff:
            best = bucket
            best_diff = diff
    return DUR_OFFSET + DUR_BUCKETS.index(best)


def token_to_duration(token: int) -> int | None:
    """Inverse of ``duration_to_token``. Returns sixteenths, or None."""
    idx = token - DUR_OFFSET
    if 0 <= idx < len(DUR_BUCKETS):
        return DUR_BUCKETS[idx]
    return None


def is_pitch_token(token: int) -> bool:
    return PITCH_OFFSET <= token < DUR_OFFSET


def is_duration_token(token: int) -> bool:
    return DUR_OFFSET <= token < VOCAB_SIZE


def is_special_token(token: int) -> bool:
    return token in SPECIAL_TOKENS
