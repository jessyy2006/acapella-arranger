"""Audio -> lead-melody tokens.

Three stages:

1. **HT-Demucs** isolates the vocal stem from a mixed-source audio file.
   HT-Demucs is a Hybrid Transformer Demucs (Rouard et al., 2023,
   arxiv.org/abs/2211.08553) that extends the original time-domain
   Demucs by adding a cross-domain Transformer between the encoder and
   decoder, attending over both time and spectrogram representations.
   We default to the ``htdemucs_ft`` checkpoint (330 MB, fine-tuned
   4-stem separation: drums, bass, other, vocals). The plain
   ``htdemucs`` (80 MB) is faster and available via ``demucs_model``
   kwarg when iteration speed matters more than stem quality; on pop
   material with dense instrumentals the fine-tuned variant is worth
   the extra ~1-2 minutes per clip because it drops far less
   background content into the vocal stem, which cascades into better
   downstream pitch tracking.

2. **torchcrepe** (PyTorch port of CREPE; Kim et al., 2018) predicts a
   pitch in Hz + confidence for every 10-ms frame of the vocal. The
   confidence value gates the rest/note decision downstream; low-
   confidence frames become rests rather than noise-driven pitch guesses.

3. **Quantisation** to the 16th-note grid using the tokenizer's
   ``duration_to_token`` helper, wrapping the result in a
   ``music21.stream.Part`` that ``encode_part`` then serialises to the
   same token format the harmony model was trained on.

Post-v1 tuning after listening verdict on `outputs/smoke_input.mp3`
(see ``docs/issues/001-soprano-divergence-and-lead-track.md``):

- **MIDI-space median filter** (wide; default ~150 ms). Replaces the
  v1 Hz-space filter, which was too narrow (50 ms) and operated before
  the int-MIDI rounding step — so scoops produced a series of chromatic
  passing tones instead of one sustained target pitch.
- **REST bridging** fills short (<=60 ms) low-confidence gaps inside a
  held note when the pitch on both sides matches. Fixes sustained notes
  fragmenting into multiple notes + rests when CREPE confidence dipped
  for a frame or two mid-note.
- **Short-run merging** absorbs any run shorter than ~80 ms into its
  longer neighbour. Collapses scoops / vibrato wobble into the target
  note the singer actually held.
- **Amplitude gate**. CREPE's confidence score tells you "is this
  signal periodic?" but not "is the singer audible?" Demucs bleed,
  reverb tails, or quiet instruments can have high periodicity and
  falsely fill a genuine pause with notes. We compute per-frame RMS
  on the isolated vocal stem (dBFS) and force REST below -50 dB.
- **Key-aware snapping** (Krumhansl-Schmuckler). The 3 fixes above
  clean frame-level jitter but can't correct genuinely chromatic pitch
  labels left over from rapid singer transitions (CREPE rounds Hz to
  the nearest semitone, and rapid ornaments cross chromatic semitones
  in physical reality). We detect the song's major/minor key from the
  post-smoothed pitch-class histogram and snap each out-of-key frame
  to the nearest in-key pitch within 2 semitones. Skips snapping when
  the best correlation is below 0.4 — safer than forcing a key on
  modulating or atonal clips.

Known remaining limitations:

- **Octave errors** from torchcrepe. Median-filter smoothing helps; we
  do NOT attempt Viterbi decoding in v1 (spec's optional upgrade).
- **Drum leakage** into the Demucs vocal stem can produce spurious
  pitches on snare transients. Run-merging covers most of this.
- **Tempo estimation** via librosa is unreliable on synthetic audio
  (the spec's test only asserts non-None output).
- **No chunking** for long audio in v1; Demucs handles ~2 minutes on
  a T4 GPU, less on CPU. For longer inputs the caller should pre-chunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Literal

import librosa
import numpy as np
import torch
import torchcrepe
from music21 import meter, note, stream
from scipy.signal import medfilt

from src.data.tokenizer import encode_part
from src.data.vocab import duration_to_token  # noqa: F401 — documents grid source

logger = logging.getLogger(__name__)

_TARGET_SR = 16_000  # torchcrepe expects 16 kHz mono
_CREPE_HOP_MS = 10.0  # 10-ms hop (torchcrepe default)
_CREPE_HOP = int(_TARGET_SR * _CREPE_HOP_MS / 1000.0)  # 160 samples
_FMIN = 50.0  # below bass low E
_FMAX = 1100.0  # above soprano high C

_CrepeModel = Literal["tiny", "small", "medium", "full"]

# Krumhansl-Schmuckler 1990 key profiles (stability weights for each scale
# degree, pitch-class 0 = tonic). Used to detect a song's key by Pearson-
# correlating these rolled profiles against the observed pitch-class
# histogram of the vocal line.
_KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)
_MAJOR_SCALE_OFFSETS: Final[tuple[int, ...]] = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE_OFFSETS: Final[tuple[int, ...]] = (0, 2, 3, 5, 7, 8, 10)  # natural minor


def _pick_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def isolate_vocals(
    audio_path: str | Path,
    device: str | torch.device | None = None,
    *,
    demucs_model: str = "htdemucs_ft",
) -> tuple[np.ndarray, int]:
    """Run HT-Demucs and return the vocal stem + sample rate.

    Loads the model on first call; the model weights are cached by
    ``demucs.pretrained.get_model`` so subsequent calls are fast.
    Returns a 2-D ``(channels, samples)`` float32 numpy array at the
    model's native sample rate (44.1 kHz).

    Default ``demucs_model='htdemucs_ft'`` is the fine-tuned variant
    (330 MB, slower, cleaner vocal stem). Switch to ``'htdemucs'``
    (80 MB, faster, rougher stem) when iteration speed matters more
    than isolation quality.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(str(audio_path))

    # Local imports — demucs pulls in heavy dependencies, don't want to
    # pay the import cost unless the caller actually needs Demucs.
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    from demucs.pretrained import get_model

    dev = _pick_device(device if isinstance(device, str) or device is None else str(device))
    logger.info("loading %s onto %s", demucs_model, dev)
    model = get_model(demucs_model)
    model.to(dev).eval()

    wav, sr = librosa.load(str(audio_path), sr=None, mono=False)
    if wav.ndim == 1:
        wav = wav[np.newaxis, :]  # (1, samples)
    tensor = torch.from_numpy(wav).float()
    tensor = convert_audio(tensor, sr, model.samplerate, model.audio_channels)
    # Demucs expects (batch, channels, samples).
    with torch.no_grad():
        stems = apply_model(model, tensor[None].to(dev), device=dev, split=True)[0]

    # Sources come back in `model.sources` order — e.g.
    # ``['drums', 'bass', 'other', 'vocals']`` for htdemucs.
    try:
        vocal_idx = model.sources.index("vocals")
    except ValueError as exc:  # pragma: no cover — shouldn't happen with htdemucs
        raise RuntimeError(f"Demucs model has no 'vocals' stem: {model.sources}") from exc
    vocal = stems[vocal_idx].cpu().numpy()
    return vocal, model.samplerate


def pitch_track(
    audio: np.ndarray,
    sample_rate: int,
    device: str | torch.device | None = None,
    *,
    model_size: _CrepeModel = "full",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run torchcrepe on ``audio`` and return frame-aligned arrays.

    Returns ``(times_sec, pitch_hz, confidence, amplitude_db)``. All four
    are 1-D arrays with one entry per 10-ms frame. ``amplitude_db`` is
    the per-frame RMS of the mono, 16 kHz input converted to dBFS; used
    downstream to gate pauses where CREPE would otherwise report a
    confident pitch on reverb tail or Demucs bleed.
    """
    dev = _pick_device(device if isinstance(device, str) or device is None else str(device))

    # Downmix + resample to the rate torchcrepe expects.
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)
    if sample_rate != _TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=_TARGET_SR)

    audio_t = torch.from_numpy(audio).float().unsqueeze(0)  # (1, samples)
    pitch, periodicity = torchcrepe.predict(
        audio_t,
        _TARGET_SR,
        hop_length=_CREPE_HOP,
        fmin=_FMIN,
        fmax=_FMAX,
        model=model_size,
        batch_size=512,
        device=dev,
        return_periodicity=True,
    )
    pitch_np = pitch[0].cpu().numpy().astype(np.float32)
    confidence_np = periodicity[0].cpu().numpy().astype(np.float32)

    # Per-frame RMS in dBFS, aligned to the CREPE frame grid. Use a
    # 20 ms analysis window (2 * hop) so the RMS picks up short bursts.
    rms = librosa.feature.rms(
        y=audio, frame_length=_CREPE_HOP * 2, hop_length=_CREPE_HOP, center=True,
    )[0]
    amplitude_db = librosa.amplitude_to_db(rms + 1e-12, ref=1.0).astype(np.float32)

    # Align lengths — librosa's centered framing sometimes emits one more
    # frame than CREPE; trim/pad to match.
    n = len(pitch_np)
    if len(amplitude_db) > n:
        amplitude_db = amplitude_db[:n]
    elif len(amplitude_db) < n:
        amplitude_db = np.pad(
            amplitude_db, (0, n - len(amplitude_db)), constant_values=amplitude_db[-1]
        )

    times = np.arange(n, dtype=np.float32) * (_CREPE_HOP / _TARGET_SR)
    return times, pitch_np, confidence_np, amplitude_db


def _hz_to_midi_int(pitch_hz: np.ndarray) -> np.ndarray:
    """Frame-wise Hz -> int MIDI, using librosa. Silent frames stay at 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        midi = librosa.hz_to_midi(np.maximum(pitch_hz, 1e-6))
    midi = np.rint(midi)
    midi = np.clip(midi, 0, 127)
    return midi.astype(np.int32)


def _bridge_short_rest_gaps(midi_frames: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill REST (-1) stretches of length <= ``max_gap`` when bounded on
    both sides by the *same* non-rest pitch.

    Fixes the "held note split by a momentary confidence dip" case — the
    singer sustains MIDI 68, CREPE confidence drops for two frames, the
    whole note otherwise fragments into two notes + a rest.
    """
    if max_gap < 1 or len(midi_frames) == 0:
        return midi_frames
    out = midi_frames.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] != -1:
            i += 1
            continue
        start = i
        while i < n and out[i] == -1:
            i += 1
        gap_len = i - start
        if (
            0 < gap_len <= max_gap
            and start > 0
            and i < n
            and out[start - 1] == out[i]
            and out[start - 1] != -1
        ):
            out[start:i] = out[start - 1]
    return out


def _merge_short_runs(
    runs: list[tuple[int, int]], min_merge_frames: int
) -> list[tuple[int, int]]:
    """Absorb any run shorter than ``min_merge_frames`` into its longer
    neighbour (tiebreak: previous).

    Handles two symptoms at once:

    - **Scoops** — a rising scoop shows up as short intermediate pitch
      runs leading into the sustained target. The target is the longer
      neighbour, so the scoop steps all collapse into it.
    - **Whole-note wobble** — vibrato that crosses a semitone boundary
      produces brief off-pitch runs inside a long held note. The held
      note is the longer neighbour, so the wobbles get absorbed.

    Iterates to a fixed point: one merge can expose another short run
    if both neighbours of a short run were themselves short.
    """
    if min_merge_frames < 2 or len(runs) < 2:
        return list(runs)
    runs = list(runs)
    changed = True
    while changed:
        changed = False
        for i in range(len(runs)):
            val, count = runs[i]
            if count >= min_merge_frames:
                continue
            if len(runs) <= 1:
                break
            if i == 0:
                nxt_val, nxt_count = runs[1]
                runs[0] = (nxt_val, count + nxt_count)
                runs.pop(1)
            elif i == len(runs) - 1:
                prev_val, prev_count = runs[-2]
                runs[-2] = (prev_val, prev_count + count)
                runs.pop()
            else:
                prev_val, prev_count = runs[i - 1]
                nxt_val, nxt_count = runs[i + 1]
                if prev_count >= nxt_count:
                    runs[i - 1] = (prev_val, prev_count + count)
                else:
                    runs[i + 1] = (nxt_val, count + nxt_count)
                runs.pop(i)
            changed = True
            break

    # A short run between two identical-pitch neighbours (e.g. a
    # 68-69-68 wobble) gets absorbed into one side, leaving the two 68
    # runs adjacent. Coalesce same-pitch neighbours so the sustained
    # note emits as one long Note, not two consecutive ones.
    coalesced: list[tuple[int, int]] = []
    for val, count in runs:
        if coalesced and coalesced[-1][0] == val:
            coalesced[-1] = (val, coalesced[-1][1] + count)
        else:
            coalesced.append((val, count))
    return coalesced


def _detect_scale_pitch_classes(
    midi_frames: np.ndarray, *, min_score: float = 0.4, min_frames: int = 20
) -> frozenset[int] | None:
    """Krumhansl-Schmuckler key detection on the pitch-class histogram.

    Returns the 7 in-key pitch classes of the best-matching major or minor
    key, or ``None`` when the signal is too sparse / the correlation is
    too weak to trust a single-key model. In the ``None`` case callers
    should leave the pitches alone rather than force-snap into a bad key.
    """
    if midi_frames.size == 0:
        return None
    valid = midi_frames[midi_frames >= 0]
    if valid.size < min_frames:
        return None

    hist = np.bincount(valid % 12, minlength=12).astype(np.float64)
    if hist.sum() <= 0:
        return None
    hist /= hist.sum()

    best_score = -np.inf
    best_pcs: frozenset[int] | None = None
    for profile, offsets in (
        (_KRUMHANSL_MAJOR, _MAJOR_SCALE_OFFSETS),
        (_KRUMHANSL_MINOR, _MINOR_SCALE_OFFSETS),
    ):
        for tonic in range(12):
            rolled = np.roll(profile, tonic)
            # np.corrcoef is NaN when a vector has zero variance — guard
            # that (all-zero histogram already returned above, but a
            # single-pitch-class histogram would also trip the check).
            if np.std(hist) == 0:
                return None
            score = float(np.corrcoef(hist, rolled)[0, 1])
            if score > best_score:
                best_score = score
                best_pcs = frozenset((tonic + o) % 12 for o in offsets)

    if best_score < min_score:
        logger.info(
            "key-detect correlation %.2f below %.2f — skipping snap",
            best_score, min_score,
        )
        return None
    return best_pcs


def _snap_to_scale(
    midi_frames: np.ndarray,
    allowed_pcs: frozenset[int],
    *,
    max_distance: int = 2,
) -> np.ndarray:
    """Snap each out-of-key, non-rest frame to the nearest in-key pitch.

    Searches outward by ``max_distance`` semitones (tries ±1, ±2, …). If
    no in-key pitch exists within the radius, the frame is left alone —
    it's a genuine accidental and the singer meant it.
    """
    if not allowed_pcs:
        return midi_frames
    out = midi_frames.copy()
    for i in range(len(out)):
        v = int(out[i])
        if v < 0 or (v % 12) in allowed_pcs:
            continue
        for delta in range(1, max_distance + 1):
            for sign in (-1, 1):
                candidate = v + sign * delta
                if 0 <= candidate <= 127 and (candidate % 12) in allowed_pcs:
                    out[i] = candidate
                    break
            else:
                continue
            break
    return out


def frames_to_part(
    times_sec: np.ndarray,
    pitch_hz: np.ndarray,
    confidence: np.ndarray,
    tempo_bpm: float,
    *,
    confidence_threshold: float = 0.5,
    amplitude_db: np.ndarray | None = None,
    amplitude_threshold_db: float = -50.0,
    amplitude_relative_db: float = 30.0,
    smoothing_window: int = 5,
    pitch_smoothing_window: int = 21,
    rest_bridge_frames: int = 6,
    min_merge_frames: int = 8,
    enable_key_snap: bool = True,
    key_snap_max_distance: int = 2,
    min_sixteenths: int = 1,
) -> stream.Part:
    """Collapse frame-level pitch into a quantised monophonic Part.

    Pipeline (per issue 001 audio-pipeline tuning):

    1. Median-filter *confidence* (Hz-level median filter is a no-op for
       our purposes — integer-MIDI smoothing below is more effective).
    2. Convert Hz -> int MIDI, mark frames as REST (-1) when confidence
       is below ``confidence_threshold`` OR amplitude is below an
       *adaptive* gate: ``max(amplitude_threshold_db, p95(amplitude_db)
       - amplitude_relative_db)``. The gate scales with the song's
       loud moments so quiet sections (softer singing, outros) use a
       stricter effective threshold than loud sections — this is the
       only way to catch reverb/bleed in pauses without also washing
       out soft high notes in the main material.
    3. **Wide median filter on MIDI integers** (``pitch_smoothing_window``,
       default ~150 ms): kills semitone-boundary flicker that makes
       scoops read as flats/sharps and subsumes 1-2 frame REST blips
       into surrounding pitches.
    4. **Explicit REST bridging** (``rest_bridge_frames``): REST gaps up
       to N frames bounded by the same pitch on both sides are filled
       with that pitch. Fixes longer confidence dips inside held notes
       that the median filter can't reach.
    5. **Key-aware snapping** (when ``enable_key_snap``): Krumhansl-
       Schmuckler detects the best-matching major/minor key from the
       pitch-class histogram, then any out-of-key frame within
       ``key_snap_max_distance`` semitones of an in-key pitch is pulled
       onto that pitch. Kills chromatic passing tones left over from
       CREPE's faithful-to-the-Hz rounding at fast transitions.
    6. Group into runs, then **merge any run shorter than
       ``min_merge_frames`` into its longer neighbour** — collapses
       scoop transitions into the target note.
    7. Emit music21 notes/rests, ``makeMeasures`` into 4/4.
    """
    if tempo_bpm <= 0:
        raise ValueError(f"tempo_bpm must be positive, got {tempo_bpm}")
    if len(times_sec) != len(pitch_hz) or len(times_sec) != len(confidence):
        raise ValueError("times_sec, pitch_hz, confidence must be same length")

    if smoothing_window >= 3 and len(confidence) >= smoothing_window:
        win = smoothing_window if smoothing_window % 2 else smoothing_window + 1
        confidence = medfilt(confidence, kernel_size=win)

    midi_frames = _hz_to_midi_int(pitch_hz)
    is_voiced = confidence >= confidence_threshold
    if amplitude_db is not None:
        if len(amplitude_db) != len(midi_frames):
            raise ValueError(
                f"amplitude_db length {len(amplitude_db)} must match pitch_hz "
                f"length {len(midi_frames)}"
            )
        # Adaptive threshold: set the gate relative to the song's loud
        # moments (95th percentile), never below the floor. A fixed
        # threshold either over-rejects soft notes (missing high falsetto)
        # OR under-rejects Demucs bleed (missing pauses) — the two
        # failure modes we saw on the smoke clip. Relative gating
        # resolves the tension because the gate scales with the actual
        # signal level.
        p95 = float(np.percentile(amplitude_db, 95))
        effective_threshold = max(amplitude_threshold_db, p95 - amplitude_relative_db)
        logger.info(
            "amp gate: p95=%.1f dB, effective threshold=%.1f dB (floor=%.1f, relative=%.1f)",
            p95, effective_threshold, amplitude_threshold_db, amplitude_relative_db,
        )
        is_voiced = is_voiced & (amplitude_db >= effective_threshold)
    midi_frames = np.where(is_voiced, midi_frames, -1).astype(np.int32)

    # Wide median filter in MIDI-integer space — the key fix for scoops.
    if pitch_smoothing_window >= 3 and len(midi_frames) >= pitch_smoothing_window:
        win = (
            pitch_smoothing_window
            if pitch_smoothing_window % 2
            else pitch_smoothing_window + 1
        )
        midi_frames = medfilt(midi_frames.astype(np.float32), kernel_size=win).astype(
            np.int32
        )

    midi_frames = _bridge_short_rest_gaps(midi_frames, rest_bridge_frames)

    if enable_key_snap:
        allowed_pcs = _detect_scale_pitch_classes(midi_frames)
        if allowed_pcs is not None:
            midi_frames = _snap_to_scale(
                midi_frames, allowed_pcs, max_distance=key_snap_max_distance
            )

    # Group consecutive frames with the same value into runs.
    runs: list[tuple[int, int]] = []  # (midi_or_-1, frame_count)
    if len(midi_frames) == 0:
        return stream.Part()
    cur_val = int(midi_frames[0])
    cur_count = 1
    for val in midi_frames[1:]:
        iv = int(val)
        if iv == cur_val:
            cur_count += 1
        else:
            runs.append((cur_val, cur_count))
            cur_val, cur_count = iv, 1
    runs.append((cur_val, cur_count))

    runs = _merge_short_runs(runs, min_merge_frames)

    # Frame hop (assume uniform) -> seconds -> sixteenths.
    frame_hop_sec = float(times_sec[1] - times_sec[0]) if len(times_sec) >= 2 else (
        _CREPE_HOP / _TARGET_SR
    )
    sixteenth_sec = 60.0 / (4.0 * tempo_bpm)

    part = stream.Part()
    part.append(meter.TimeSignature("4/4"))
    for midi_val, frame_count in runs:
        duration_sec = frame_count * frame_hop_sec
        sixteenths = int(round(duration_sec / sixteenth_sec))
        if sixteenths < min_sixteenths:
            continue  # drop spurious blips (likely drum leakage)
        quarter_length = sixteenths / 4.0
        if midi_val < 0:
            part.append(note.Rest(quarterLength=quarter_length))
        else:
            part.append(note.Note(int(midi_val), quarterLength=quarter_length))

    # Re-measure into 4/4 so encode_part emits BAR tokens. Fall back to a
    # flat Part if makeMeasures fails (empty Part, etc.).
    try:
        measured = part.makeMeasures()
    except Exception:  # pragma: no cover — belt-and-braces
        return part
    return measured


def extract_lead_tokens(
    audio_path: str | Path,
    *,
    tempo_bpm: float | None = None,
    device: str | torch.device | None = None,
    crepe_model: _CrepeModel = "full",
    confidence_threshold: float = 0.5,
    demucs_model: str = "htdemucs_ft",
) -> list[int]:
    """End-to-end: audio file -> lead-melody token list.

    Chains :func:`isolate_vocals`, :func:`pitch_track`,
    :func:`frames_to_part`, and the tokenizer's :func:`encode_part`.
    If ``tempo_bpm`` is ``None``, estimates tempo with
    ``librosa.beat.beat_track`` on the isolated vocal.

    Returns a token list in the same vocabulary the harmony model was
    trained on — framed by SOS/EOS, BAR-delimited measures, REST tokens
    for silent / low-confidence regions.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(str(audio_path))

    logger.info("isolating vocals from %s", audio_path)
    vocal, sr = isolate_vocals(audio_path, device=device, demucs_model=demucs_model)

    if tempo_bpm is None:
        mono = librosa.to_mono(vocal) if vocal.ndim > 1 else vocal
        tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
        tempo_bpm = float(tempo) if np.isscalar(tempo) or tempo.ndim == 0 else float(tempo[0])
        if tempo_bpm <= 0:
            logger.warning("librosa returned tempo=%.2f; falling back to 120", tempo_bpm)
            tempo_bpm = 120.0
        logger.info("estimated tempo: %.1f bpm", tempo_bpm)

    logger.info("pitch-tracking with torchcrepe (model=%s)", crepe_model)
    times, pitch_hz, conf, amp_db = pitch_track(
        vocal, sr, device=device, model_size=crepe_model
    )

    part = frames_to_part(
        times, pitch_hz, conf, tempo_bpm,
        confidence_threshold=confidence_threshold,
        amplitude_db=amp_db,
    )
    tokens = encode_part(part)
    logger.info("extracted %d tokens from %s", len(tokens), audio_path.name)
    return tokens
