"""Low-level loaders shared by the tokenizer, dataset, and exploration notebooks.

Promoted from ``notebooks/01_data_exploration.ipynb`` so downstream code
(``tokenizer.py``, ``dataset.py``, etc.) has a single import source.
"""

from __future__ import annotations

import re
from pathlib import Path

from music21 import converter, stream

SATB_NAMES: frozenset[str] = frozenset({"Soprano", "Alto", "Tenor", "Bass"})

_DOCTYPE_RE = re.compile(r"<!DOCTYPE[^>]*>\s*")


def parse_lenient(xml_path: str | Path) -> stream.Score:
    """Parse a jaCappella MusicXML file, stripping the MuseScore-4 DOCTYPE.

    music21's default parser resolves the external DTD referenced in the
    DOCTYPE and fails without network access. The XML body itself is
    well-formed, so we strip the DOCTYPE and hand the content to music21.
    """
    text = Path(xml_path).read_text(encoding="utf-8")
    text = _DOCTYPE_RE.sub("", text)
    return converter.parseData(text, format="musicxml")


def is_clean_satb(score: stream.Score) -> bool:
    """True iff the score has exactly four parts named Soprano/Alto/Tenor/Bass.

    Used to filter the music21 JSB Chorales corpus, which mixes pure
    four-voice chorales with cantata fragments that add Oboe / Violin /
    Continuo / Trumpet parts we can't use for SATB training.
    """
    names = [(p.partName or "").strip() for p in score.parts]
    return len(names) == 4 and set(names) == SATB_NAMES
