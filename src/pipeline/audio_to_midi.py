"""Audio -> lead-melody tokens.

Three stages:

1. **HT-Demucs** isolates the vocal stem from a mixed-source audio file.
   HT-Demucs is a Hybrid Transformer Demucs (Rouard et al., 2023,
   arxiv.org/abs/2211.08553) that extends the original time-domain
   Demucs by adding a cross-domain Transformer between the encoder and
   decoder, attending over both time and spectrogram representations.
   We use the ``htdemucs`` checkpoint (80 MB, 4-stem separation: drums,
   bass, other, vocals) — index 3 of the output is the vocal stem.

2. **torchcrepe** (PyTorch port of CREPE; Kim et al., 2018) predicts a
   pitch in Hz + confidence for every 10-ms frame of the vocal. The
   confidence value gates the rest/note decision downstream; low-
   confidence frames become rests rather than noise-driven pitch guesses.

3. **Quantisation** to the 16th-note grid using the tokenizer's
   ``duration_to_token`` helper, wrapping the result in a
   ``music21.stream.Part`` that ``encode_part`` then serialises to the
   same token format the harmony model was trained on.

Known limitations (carried into the eval write-up):

- **Octave errors** from torchcrepe. Median-filter smoothing helps; we
  do NOT attempt Viterbi decoding in v1 (spec's optional upgrade).
- **Drum leakage** into the Demucs vocal stem can produce spurious
  pitches on snare transients. A minimum-note-duration filter drops the
  shortest runs as a mitigation.
- **Talking / non-pitched segments** degrade gracefully to REST tokens
  because confidence drops below threshold.
- **Tempo estimation** via librosa is unreliable on synthetic audio
  (the spec's test only asserts non-None output).
- **No chunking** for long audio in v1; Demucs handles ~2 minutes on
  a T4 GPU, less on CPU. For longer inputs the caller should pre-chunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

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
) -> tuple[np.ndarray, int]:
    """Run HT-Demucs and return the vocal stem + sample rate.

    Loads the model on first call; the model weights are cached by
    ``demucs.pretrained.get_model`` so subsequent calls are fast.
    Returns a 2-D ``(channels, samples)`` float32 numpy array at the
    model's native sample rate (44.1 kHz).
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
    logger.info("loading htdemucs onto %s", dev)
    model = get_model("htdemucs")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run torchcrepe on ``audio``, returning ``(times_sec, pitch_hz, confidence)``.

    The input is resampled to 16 kHz mono internally; the three returned
    arrays are aligned 1-D arrays with one entry per 10-ms frame.
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
    times = np.arange(len(pitch_np), dtype=np.float32) * (_CREPE_HOP / _TARGET_SR)
    return times, pitch_np, confidence_np


def _hz_to_midi_int(pitch_hz: np.ndarray) -> np.ndarray:
    """Frame-wise Hz -> int MIDI, using librosa. Silent frames stay at 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        midi = librosa.hz_to_midi(np.maximum(pitch_hz, 1e-6))
    midi = np.rint(midi)
    midi = np.clip(midi, 0, 127)
    return midi.astype(np.int32)


def frames_to_part(
    times_sec: np.ndarray,
    pitch_hz: np.ndarray,
    confidence: np.ndarray,
    tempo_bpm: float,
    *,
    confidence_threshold: float = 0.5,
    smoothing_window: int = 5,
    min_sixteenths: int = 1,
) -> stream.Part:
    """Collapse frame-level pitch into a quantised monophonic Part.

    Groups runs of consecutive frames with the same median-filtered MIDI
    pitch into notes (or rests for sub-threshold confidence). Durations
    are expressed in quarter-lengths so music21 can later re-measure the
    stream into 4/4 measures via ``makeMeasures``.
    """
    if tempo_bpm <= 0:
        raise ValueError(f"tempo_bpm must be positive, got {tempo_bpm}")
    if len(times_sec) != len(pitch_hz) or len(times_sec) != len(confidence):
        raise ValueError("times_sec, pitch_hz, confidence must be same length")

    # Median-filter pitch + confidence to damp torchcrepe jitter + octave errors.
    if smoothing_window >= 3 and len(pitch_hz) >= smoothing_window:
        win = smoothing_window if smoothing_window % 2 else smoothing_window + 1
        pitch_hz = medfilt(pitch_hz, kernel_size=win)
        confidence = medfilt(confidence, kernel_size=win)

    midi_frames = _hz_to_midi_int(pitch_hz)
    # Mark low-confidence frames as rest (-1 sentinel).
    midi_frames = np.where(confidence >= confidence_threshold, midi_frames, -1).astype(
        np.int32
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
    vocal, sr = isolate_vocals(audio_path, device=device)

    if tempo_bpm is None:
        mono = librosa.to_mono(vocal) if vocal.ndim > 1 else vocal
        tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
        tempo_bpm = float(tempo) if np.isscalar(tempo) or tempo.ndim == 0 else float(tempo[0])
        if tempo_bpm <= 0:
            logger.warning("librosa returned tempo=%.2f; falling back to 120", tempo_bpm)
            tempo_bpm = 120.0
        logger.info("estimated tempo: %.1f bpm", tempo_bpm)

    logger.info("pitch-tracking with torchcrepe (model=%s)", crepe_model)
    times, pitch_hz, conf = pitch_track(vocal, sr, device=device, model_size=crepe_model)

    part = frames_to_part(
        times, pitch_hz, conf, tempo_bpm,
        confidence_threshold=confidence_threshold,
    )
    tokens = encode_part(part)
    logger.info("extracted %d tokens from %s", len(tokens), audio_path.name)
    return tokens
