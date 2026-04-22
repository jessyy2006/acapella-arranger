"""Section detection + token-window mapping for section-aware arrangement.

Implements the first step of ``docs/issues/003-satb-voices-lack-
sectional-pattern-structure.md``: segment the lead audio into
verse / chorus / bridge style labels so ``run_pipeline`` can generate
each unique section once and paste the output across repeats.

Two public helpers:

- :func:`detect_sections` — audio → ``[(label, start_sec, end_sec), ...]``,
  contiguous, covering ``[0, duration]``. Labels repeat across similar
  segments (e.g. ``A`` for both verses).
- :func:`tokens_for_section_window` — map a ``[start_sec, end_sec]``
  window onto a slice of the lead token stream using the tokenizer's
  16th-note grid and an assumed tempo.

Both helpers degrade gracefully when segmentation fails or the input is
too short: :func:`detect_sections` returns a single-section fallback,
and :func:`tokens_for_section_window` returns an empty range.
"""

from __future__ import annotations

import logging
from typing import Final

import librosa
import numpy as np

from src.data.vocab import (
    BAR,
    EOS,
    PAD,
    REST,
    SOS,
    is_duration_token,
    is_pitch_token,
    token_to_duration,
)

logger = logging.getLogger(__name__)

# Segmentation defaults. Shorter-than-min clips collapse to one section;
# longer clips get split into up to ``_MAX_SEGMENTS`` and clustered into
# at most ``_MAX_LABELS`` labels. The minimum section length is set to
# 10 s so patterns hold long enough to read as "verse / chorus" rather
# than as two-bar ornaments — a shorter floor produced sectional labels
# that flipped every few bars on the smoke clip.
_MIN_SECTION_SEC: Final[float] = 10.0
_MAX_SEGMENTS: Final[int] = 12
_MAX_LABELS: Final[int] = 5


def _fallback_single_section(duration_sec: float) -> list[tuple[str, float, float]]:
    return [("A", 0.0, max(0.0, float(duration_sec)))]


def _label_sequence(cluster_ids: list[int]) -> list[str]:
    """Map integer cluster ids to ``"A"``, ``"B"``, ... in first-appearance
    order so ``[2, 0, 2, 1]`` → ``["A", "B", "A", "C"]``.
    """
    seen: dict[int, str] = {}
    out: list[str] = []
    for cid in cluster_ids:
        if cid not in seen:
            seen[cid] = chr(ord("A") + len(seen))
        out.append(seen[cid])
    return out


def detect_sections(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_section_sec: float = _MIN_SECTION_SEC,
    max_segments: int = _MAX_SEGMENTS,
    max_labels: int = _MAX_LABELS,
) -> list[tuple[str, float, float]]:
    """Detect section boundaries + assign repetition labels.

    Parameters
    ----------
    audio
        1-D or multi-channel audio array. Multi-channel is downmixed to
        mono internally.
    sample_rate
        Hz.

    Returns
    -------
    ``[(label, start_sec, end_sec), ...]`` covering ``[0, duration]``
    with no gaps or overlaps. Labels are strings like ``"A"`` / ``"B"``;
    identical labels indicate mutually similar sections. Minimum of
    one entry is always returned.
    """
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)
    duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

    if duration < 2 * min_section_sec:
        return _fallback_single_section(duration)

    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        # Pick segment count from audio length; floor of 2, cap at max_segments.
        k_segments = max(2, min(max_segments, int(duration / min_section_sec)))
        bounds = librosa.segment.agglomerative(mfcc, k=k_segments)
        # ``bounds`` is an array of segment-start frame indices (k of them,
        # first is always 0). Append the final frame to close the last
        # segment, then convert frame indices to seconds.
        start_frames = [int(b) for b in bounds]
        start_frames.append(int(mfcc.shape[1]))
        starts_sec = [float(t) for t in librosa.frames_to_time(start_frames, sr=sample_rate)]
        # Clamp the final end to the audio duration — frames_to_time can
        # overshoot by one hop.
        starts_sec[-1] = duration

        segments: list[tuple[float, float]] = []
        for i in range(len(starts_sec) - 1):
            s, e = starts_sec[i], starts_sec[i + 1]
            if e - s >= min_section_sec * 0.5:  # drop near-empty segments
                segments.append((s, e))

        if len(segments) < 2:
            return _fallback_single_section(duration)

        # Close any micro-gaps left by dropped near-empty segments — the
        # downstream token slicer expects contiguous coverage, and leaving
        # sub-frame gaps between sections causes the last couple frames
        # of each section to vanish when re-stitched.
        closed: list[tuple[float, float]] = [segments[0]]
        for s, e in segments[1:]:
            closed.append((closed[-1][1], e))
        # Make sure we still cover to ``duration`` exactly.
        closed[-1] = (closed[-1][0], duration)
        segments = closed

        # Summarise each segment with its mean MFCC vector, then cluster
        # the summaries to assign repetition labels.
        seg_features: list[np.ndarray] = []
        for s, e in segments:
            sf = int(librosa.time_to_frames(s, sr=sample_rate))
            ef = int(librosa.time_to_frames(e, sr=sample_rate))
            ef = max(ef, sf + 1)  # ensure non-empty slice
            seg_features.append(mfcc[:, sf:ef].mean(axis=1))
        feats = np.stack(seg_features, axis=0)

        # Aim for roughly half as many labels as segments so at least
        # one label repeats — the whole point of the section-aware path
        # is to give the listener audible motif repetition (e.g.,
        # "A, B, A, B" or "A, A, B, B"). Floor 2, cap at ``max_labels``.
        n_clusters = max(2, min(max_labels, len(segments) // 2))
        if n_clusters >= len(segments):
            # All segments distinct — nothing to cluster.
            cluster_ids = list(range(len(segments)))
        elif n_clusters < 2:
            cluster_ids = [0] * len(segments)
        else:
            # Local import — sklearn is heavy, only pay the cost when we
            # actually cluster.
            from sklearn.cluster import AgglomerativeClustering

            clust = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_ids = clust.fit_predict(feats).tolist()

        labels = _label_sequence(cluster_ids)
        result = [(labels[i], segments[i][0], segments[i][1]) for i in range(len(segments))]
        logger.info(
            "detected %d section(s) with %d unique label(s): %s",
            len(result), len(set(labels)),
            [(lbl, round(s, 2), round(e, 2)) for lbl, s, e in result],
        )
        return result
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("section detection failed (%s); falling back to single section", exc)
        return _fallback_single_section(duration)


def tokens_for_section_window(
    tokens: list[int],
    tempo_bpm: float,
    start_sec: float,
    end_sec: float,
) -> tuple[int, int]:
    """Map a ``[start_sec, end_sec]`` window onto a contiguous token slice.

    Walks ``tokens`` summing duration buckets (converted to seconds via
    ``tempo_bpm``) and returns ``(start_idx, end_idx)`` such that
    ``tokens[start_idx:end_idx]`` covers the requested window.

    The slice **does not** include the surrounding SOS/EOS — callers
    that want a well-formed sub-sequence should re-wrap. An empty slice
    (``start_idx == end_idx``) is returned if the window falls outside
    the token stream's total duration.
    """
    if tempo_bpm <= 0 or start_sec >= end_sec:
        return (0, 0)

    sixteenth_sec = 60.0 / (4.0 * tempo_bpm)
    # Time in seconds accumulated across the stream.
    t = 0.0
    start_idx = -1
    end_idx = -1

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in (SOS, EOS, BAR, PAD):
            if tok == EOS:
                break
            i += 1
            continue
        if is_pitch_token(tok) or tok == REST:
            if i + 1 >= len(tokens) or not is_duration_token(tokens[i + 1]):
                i += 1
                continue
            dur_sixteenths = token_to_duration(tokens[i + 1]) or 0
            pair_sec = dur_sixteenths * sixteenth_sec
            if start_idx < 0 and t + pair_sec > start_sec:
                start_idx = i
            t += pair_sec
            if end_idx < 0 and t >= end_sec:
                end_idx = i + 2
                break
            i += 2
            continue
        i += 1

    if start_idx < 0:
        return (0, 0)
    if end_idx < 0:
        end_idx = len(tokens)
    return (start_idx, end_idx)
