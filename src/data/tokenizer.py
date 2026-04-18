"""Score <-> token-sequence round-trip on the ``vocab`` grammar.

A sequence looks like::

    [SOS, PITCH, DUR, PITCH, DUR, BAR, PITCH, DUR, ..., EOS]

Each note contributes one ``(PITCH | REST, DUR)`` pair; ``BAR`` marks a
measure boundary. The format is intentionally dumb — the Transformer+LSTM
hybrid downstream will do the heavy lifting.

Known limitations (carried forward to the eval section of the final README)
---------------------------------------------------------------------------
* **Lossy duration quantisation.** Every note is snapped to one of eight
  16th-grid buckets ``(1, 2, 3, 4, 6, 8, 12, 16)``. Tuplets (e.g. a Bach
  8th-note triplet) and dotted-16ths lose their exact rhythm; they become
  the nearest-duration straight note. Ties are collapsed into whatever
  quantised duration each notehead lands on rather than being summed.
  This is intentional for v1: the generative model only needs a
  well-defined discrete output space, not millisecond-accurate rhythm.
  Call this out in the eval write-up and measure its impact via
  round-trip duration-error statistics when we run the ablation study.
"""

from __future__ import annotations

import logging
from typing import Iterable

from music21 import chord, note, stream

from src.data.vocab import (
    BAR,
    DUR_BUCKETS,
    EOS,
    REST,
    SOS,
    duration_to_token,
    is_duration_token,
    is_pitch_token,
    pitch_to_token,
    token_to_duration,
    token_to_pitch,
)

logger = logging.getLogger(__name__)

# music21 stores durations as quarterLength floats; our grid is 16ths.
_SIXTEENTHS_PER_QUARTER = 4


def _quarter_to_sixteenths(quarter_length: float) -> float:
    return quarter_length * _SIXTEENTHS_PER_QUARTER


def _sixteenths_to_quarter(sixteenths: int) -> float:
    return sixteenths / _SIXTEENTHS_PER_QUARTER


def encode_part(
    part: stream.Part,
    *,
    include_sos_eos: bool = True,
) -> list[int]:
    """Serialise a monophonic music21 Part to a list of token ids.

    Measures are separated by ``BAR`` tokens. If the Part has no explicit
    measures (e.g. a flat stream), the output omits ``BAR`` entirely.
    """
    tokens: list[int] = []
    if include_sos_eos:
        tokens.append(SOS)

    measures = list(part.getElementsByClass(stream.Measure))
    if measures:
        for i, measure in enumerate(measures):
            if i > 0:
                tokens.append(BAR)
            tokens.extend(_encode_elements(measure.notesAndRests))
    else:
        tokens.extend(_encode_elements(part.flatten().notesAndRests))

    if include_sos_eos:
        tokens.append(EOS)
    return tokens


def _encode_elements(elements: Iterable[note.GeneralNote]) -> list[int]:
    """Encode a sequence of Notes/Rests/Chords into interleaved token pairs."""
    out: list[int] = []
    for el in elements:
        sixteenths = _quarter_to_sixteenths(float(el.quarterLength))
        if sixteenths <= 0:
            # Grace notes / zero-length artifacts don't survive quantisation.
            logger.debug("skipping zero-duration element: %r", el)
            continue

        if isinstance(el, note.Note):
            out.append(pitch_to_token(el.pitch.midi))
        elif isinstance(el, note.Rest):
            out.append(REST)
        elif isinstance(el, chord.Chord):
            # SATB voices are monophonic by construction; a chord here means
            # the source score collapsed a line. Fall back to the top note.
            top = max(el.pitches, key=lambda p: p.midi)
            logger.warning(
                "chord in monophonic part, keeping top note %s", top.nameWithOctave
            )
            out.append(pitch_to_token(top.midi))
        else:
            logger.debug("skipping unsupported element: %r", el)
            continue

        out.append(duration_to_token(sixteenths))
    return out


def decode_part(tokens: list[int]) -> stream.Part:
    """Inverse of ``encode_part``. Returns a music21 Part with measures.

    Robust to the usual garbage: unknown tokens are dropped with a warning,
    pitch tokens without a following duration are discarded.
    """
    part = stream.Part()
    current_measure = stream.Measure()

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == SOS:
            i += 1
            continue
        if tok == EOS:
            break
        if tok == BAR:
            part.append(current_measure)
            current_measure = stream.Measure()
            i += 1
            continue

        if is_pitch_token(tok) or tok == REST:
            if i + 1 >= len(tokens) or not is_duration_token(tokens[i + 1]):
                logger.warning(
                    "pitch/rest token at index %d not followed by duration; dropping",
                    i,
                )
                i += 1
                continue

            sixteenths = token_to_duration(tokens[i + 1])
            assert sixteenths is not None  # guarded by is_duration_token
            quarter_length = _sixteenths_to_quarter(sixteenths)

            if tok == REST:
                element: note.GeneralNote = note.Rest(quarterLength=quarter_length)
            else:
                midi = token_to_pitch(tok)
                assert midi is not None
                element = note.Note(midi, quarterLength=quarter_length)

            current_measure.append(element)
            i += 2
            continue

        logger.warning("unknown token %d at index %d; dropping", tok, i)
        i += 1

    if len(current_measure.notesAndRests) > 0:
        part.append(current_measure)
    return part


def is_valid_duration_sixteenths(sixteenths: int) -> bool:
    """True iff ``sixteenths`` is exactly one of our grid buckets."""
    return sixteenths in DUR_BUCKETS
