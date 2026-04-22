"""End-to-end inference: audio file -> SATB arrangement MIDI.

Chains:

1. :func:`src.pipeline.audio_to_midi.extract_lead_tokens` — HT-Demucs
   vocal isolation, torchcrepe pitch-tracking, 16th-grid quantisation,
   tokenization.
2. A trained harmony model (``SATBHybrid`` by default) generates the
   four SATB voice streams autoregressively via
   :func:`src.pipeline.decode.generate_voice_tokens` — uses the locked-
   in sampling config (pitch temperature 0.5, duration temperature 1.1,
   top-k 10) because greedy decoding produces degenerate inner voices.
3. Optional voice-leading post-process via
   :func:`src.postprocess.voice_leading.apply_voice_leading` (range-
   clamp + parallel-motion logging).
4. :func:`src.data.tokenizer.decode_part` turns each voice's tokens
   back into a ``music21.stream.Part``; the four Parts stack into a
   Score and write as MIDI.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Final

import librosa
import mido
import torch
from music21 import stream, tempo

from src.data.tokenizer import decode_part
from src.data.vocab import (
    BAR,
    EOS,
    PAD,
    REST,
    SOS,
    is_duration_token,
    is_pitch_token,
    token_to_duration,
    token_to_pitch,
)
from src.eval.evaluate import _build_model, _load_hparams_from_sources, _resolve_device
from src.pipeline.audio_to_midi import extract_lead_tokens
from src.pipeline.decode import VOICES, generate_voice_tokens
from src.pipeline.sections import detect_sections, tokens_for_section_window
from src.postprocess.voice_leading import apply_voice_leading, fit_to_length

logger = logging.getLogger(__name__)


def _seed_for_label(label: str) -> int:
    """Stable 31-bit seed derived from a section label. Identical labels
    across the same session produce identical generate_voice_tokens
    output — the only knob the section-aware path uses to guarantee
    "pattern A repeats."
    """
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def _strip_framing(tokens: list[int]) -> list[int]:
    """Drop leading SOS / trailing EOS so per-section streams can be
    concatenated without producing illegal tokens in the middle of a
    stitched sequence.
    """
    start = 1 if tokens and tokens[0] == SOS else 0
    end = len(tokens) - 1 if tokens and tokens[-1] == EOS else len(tokens)
    return tokens[start:end]


def _assemble_score(
    lead_tokens: list[int],
    generated: dict[str, list[int]],
    *,
    tempo_bpm: float | None = None,
) -> stream.Score:
    """Stack Lead + S/A/T/B into one Score with Lead on the top staff.

    Split out so tests can assert on the part structure in-memory —
    music21's MIDI writer does not preserve ``partName`` on round-trip,
    so file-level assertions cannot check staff order.

    When ``tempo_bpm`` is given it is written as a ``MetronomeMark`` at
    offset 0 so the exported MIDI plays back at the tempo used during
    quantisation — without it, music21 defaults to 120 BPM and any
    tempo != 120 skews the nominal duration of the file. Omitted in
    ``partName``-only tests that don't care about playback tempo.
    """
    score = stream.Score()
    if tempo_bpm is not None:
        score.insert(0, tempo.MetronomeMark(number=tempo_bpm))
    lead_part = decode_part(lead_tokens)
    lead_part.partName = "LEAD"
    score.insert(0, lead_part)
    for voice in VOICES:
        part = decode_part(generated[voice])
        part.partName = voice.upper()
        score.insert(0, part)
    return score


def _tokens_to_note_events(
    tokens: list[int],
) -> tuple[list[tuple[int, int, int]], int]:
    """Walk a voice's token stream and emit ``(start_16, end_16, pitch)``
    tuples in 16th-note units plus the total sixteenth count. Rest
    events advance time without emitting a note.
    """
    events: list[tuple[int, int, int]] = []
    t_sx = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == EOS:
            break
        if tok in (SOS, BAR, PAD):
            i += 1
            continue
        if is_pitch_token(tok) or tok == REST:
            if i + 1 >= len(tokens) or not is_duration_token(tokens[i + 1]):
                i += 1
                continue
            dur_sx = token_to_duration(tokens[i + 1]) or 0
            end = t_sx + dur_sx
            if is_pitch_token(tok):
                midi = token_to_pitch(tok)
                if midi is not None:
                    events.append((t_sx, end, int(midi)))
            t_sx = end
            i += 2
            continue
        i += 1
    return events, t_sx


# General MIDI program 0 "Acoustic Grand Piano" for every track.
# Piano is in the default GM sound bank every DAW ships with, so the
# file opens in GarageBand / Logic / etc. without a "download extra
# sounds" prompt. The choir-patch programs (52 / 53) triggered exactly
# that prompt in GarageBand because the DLS sound set swaps them for
# downloadable voices.
_PROGRAM_LEAD: Final[int] = 0
_PROGRAM_SATB: Final[int] = 0
_MIDI_PPQ: Final[int] = 960  # ticks per quarter note — 960 is divisible by 4, 6, 8, 16
_MIDI_VELOCITY: Final[int] = 80


def _write_exact_midi(
    lead_tokens: list[int],
    generated: dict[str, list[int]],
    *,
    tempo_bpm: float,
    target_sixteenths: int,
    out_path: Path,
) -> None:
    """Write Lead + S/A/T/B to a MIDI file with exact 16th-note timing.

    Uses ``mido`` directly so the tempo microseconds-per-quarter written
    into the file is computed from ``tempo_bpm`` with no rounding
    surprises — ``pretty_midi`` re-derives tempo from an internal tick
    scale and can drift a few hundred microseconds, which compounds to
    double-digit-ms overshoot on a 60-second clip. With ``_MIDI_PPQ``
    divisible by every :data:`~src.data.vocab.DUR_BUCKETS` value, every
    note lands on an exact integer tick.
    """
    ticks_per_sixteenth = _MIDI_PPQ // 4
    target_ticks = target_sixteenths * ticks_per_sixteenth
    # round() because tempo_bpm is derived from a float input duration /
    # target-sixteenths division; the integer microseconds-per-quarter
    # is what actually gets written into the MIDI file.
    us_per_quarter = int(round(60_000_000.0 / tempo_bpm))

    mid = mido.MidiFile(ticks_per_beat=_MIDI_PPQ)

    # Track 0: tempo + meta only. mido puts the tempo meta on its own
    # conductor track by convention; every DAW / player reads it from
    # there regardless of which instrument track follows. The
    # end_of_track lands on ``target_ticks`` so the conductor track
    # itself has the full duration even though it has no notes.
    meta_track = mido.MidiTrack()
    meta_track.append(mido.MetaMessage("set_tempo", tempo=us_per_quarter, time=0))
    meta_track.append(mido.MetaMessage("end_of_track", time=target_ticks))
    mid.tracks.append(meta_track)

    def _append_voice(name: str, tokens: list[int], program: int) -> None:
        events, _total_sx = _tokens_to_note_events(tokens)
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        track.append(mido.Message("program_change", program=program, time=0))
        # mido uses delta ticks; interleave note_on / note_off in time
        # order so deltas compute cleanly.
        timed: list[tuple[int, str, int]] = []
        for start_sx, end_sx, pitch in events:
            timed.append((start_sx * ticks_per_sixteenth, "on", pitch))
            timed.append((end_sx * ticks_per_sixteenth, "off", pitch))
        timed.sort(key=lambda x: (x[0], 0 if x[1] == "off" else 1))
        prev = 0
        for abs_tick, kind, pitch in timed:
            delta = abs_tick - prev
            if kind == "on":
                track.append(mido.Message("note_on", note=pitch, velocity=_MIDI_VELOCITY, time=delta))
            else:
                track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
            prev = abs_tick
        # Extend the track to exactly ``target_ticks`` even when the
        # voice's last note ended earlier (fit_to_length's trailing
        # REST pads time but emits no note events). Without this, a
        # voice that got padded would end before the other tracks and
        # a DAW would show a shorter part.
        tail_delta = max(0, target_ticks - prev)
        track.append(mido.MetaMessage("end_of_track", time=tail_delta))
        mid.tracks.append(track)

    _append_voice("LEAD", lead_tokens, _PROGRAM_LEAD)
    for voice in VOICES:
        _append_voice(voice.upper(), generated[voice], _PROGRAM_SATB)

    mid.save(str(out_path))


def _load_model(
    checkpoint_path: Path, model_class: str, device: torch.device
) -> torch.nn.Module:
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    hparams = _load_hparams_from_sources(checkpoint_path, state, model_class)
    if not hparams:
        raise ValueError(f"no hparams resolved for {checkpoint_path}")
    model = _build_model(model_class, hparams).to(device).eval()
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    return model


def run_pipeline(
    audio_path: str | Path,
    *,
    model_checkpoint: str | Path,
    out_path: str | Path | None = None,
    voice_leading: bool = True,
    model_class: str = "hybrid",
    temperature: float = 0.5,
    duration_temperature: float = 1.1,
    top_k: int = 10,
    max_len: int = 256,
    tempo_bpm: float | None = None,
    device: str | None = None,
    # Per-voice sampling overrides. The global ``temperature`` and
    # ``duration_temperature`` defaults were tuned against chorale /
    # a cappella listening tests; on pop material each voice wants a
    # stricter-than-global regime. See ``docs/issues/002`` (bass) and
    # ``docs/issues/004`` (S/A/T). Bass wants the tightest settings
    # because bass convention is "hold the root"; upper voices are
    # slightly looser so they still have rhythmic motion.
    bass_temperature: float = 0.3,
    bass_duration_temperature: float = 0.7,
    satb_upper_temperature: float = 0.35,
    satb_upper_duration_temperature: float = 0.8,
) -> Path:
    """Arrange ``audio_path`` as an SATB MIDI. Returns the output path.

    Parameters mirror the CLI. Sampling defaults are the values locked
    in after Phase 1 listening tests; change them at your own risk. The
    bass voice uses stricter defaults than S/A/T — see the kwarg
    comment above for the rationale.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(str(audio_path))
    checkpoint_path = Path(model_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))

    dev = _resolve_device(device)
    logger.info("device=%s", dev)

    lead_tokens, tempo_used = extract_lead_tokens(
        audio_path, tempo_bpm=tempo_bpm, device=device,
    )
    if len(lead_tokens) < 3:
        raise RuntimeError(
            f"extracted only {len(lead_tokens)} tokens from {audio_path.name}; "
            "audio may be silent or too short for pitch tracking"
        )
    logger.info("lead tokens: %d (tempo=%.1f bpm)", len(lead_tokens), tempo_used)

    # Section detection runs on the original mix (instrumental density
    # helps delineate verse/chorus). Gracefully degrades to a single
    # section when detection fails or the clip is too short.
    audio_for_sections, sr_for_sections = librosa.load(str(audio_path), sr=None, mono=True)
    # Input audio duration in seconds — the ground truth every output
    # part must match after the final fit step below.
    target_duration_sec = (
        len(audio_for_sections) / sr_for_sections if sr_for_sections > 0 else 0.0
    )
    target_sixteenths = max(1, round(target_duration_sec * tempo_used * 4.0 / 60.0))
    logger.info(
        "input duration: %.3f s @ %.1f bpm -> target %d sixteenths",
        target_duration_sec, tempo_used, target_sixteenths,
    )
    sections = detect_sections(audio_for_sections, sr_for_sections)
    unique_labels = sorted({label for label, _, _ in sections})
    logger.info(
        "sections: %d total, %d unique label(s): %s",
        len(sections), len(unique_labels),
        [(lbl, round(s, 1), round(e, 1)) for lbl, s, e in sections],
    )

    model = _load_model(checkpoint_path, model_class, dev)

    # Generate SATB once per unique label (using that label's first
    # occurrence as the conditioning lead), then paste the cached
    # tokens across every occurrence of the same label. This is the
    # "verse 1 == verse 2" primitive the user asked for.
    def _v_temp(voice: str) -> tuple[float, float]:
        # Bass gets its own tighter config; S/A/T share a single
        # upper-voice override. The legacy ``temperature`` /
        # ``duration_temperature`` kwargs remain callable but no
        # longer drive per-voice sampling — callers that want to
        # widen the upper voices should raise
        # ``satb_upper_temperature`` / ``satb_upper_duration_temperature``
        # directly.
        if voice == "b":
            return (bass_temperature, bass_duration_temperature)
        return (satb_upper_temperature, satb_upper_duration_temperature)

    generated_by_label: dict[str, dict[str, list[int]]] = {}
    for label in unique_labels:
        first_start, first_end = next((s, e) for lbl, s, e in sections if lbl == label)
        slice_start, slice_end = tokens_for_section_window(
            lead_tokens, tempo_used, first_start, first_end,
        )
        section_tokens = lead_tokens[slice_start:slice_end] if slice_end > slice_start else lead_tokens
        # Re-frame with SOS/EOS so the model sees a well-formed sequence.
        if not section_tokens or section_tokens[0] != SOS:
            section_tokens = [SOS] + section_tokens
        if section_tokens[-1] != EOS:
            section_tokens = section_tokens + [EOS]
        section_tensor = torch.tensor([section_tokens], dtype=torch.long, device=dev)
        seed = _seed_for_label(label)
        gen_for_label: dict[str, list[int]] = {}
        for voice in VOICES:
            v_temp, v_dur_temp = _v_temp(voice)
            gen_for_label[voice] = generate_voice_tokens(
                model, section_tensor, voice, max_len, dev,
                temperature=v_temp,
                duration_temperature=v_dur_temp,
                top_k=top_k,
                seed=seed,
            )
        generated_by_label[label] = gen_for_label
        logger.info(
            "label %s: %s", label,
            {v: len(gen_for_label[v]) for v in VOICES},
        )

    # Stitch per-voice streams in section order. Drop interior SOS/EOS
    # so the concatenated output has one SOS at the start and one EOS
    # at the end.
    generated: dict[str, list[int]] = {v: [SOS] for v in VOICES}
    for label, _, _ in sections:
        for voice in VOICES:
            generated[voice].extend(_strip_framing(generated_by_label[label][voice]))
    for voice in VOICES:
        generated[voice].append(EOS)
        logger.info("voice %s: %d tokens (stitched)", voice, len(generated[voice]))

    if voice_leading:
        generated = apply_voice_leading(
            generated, enable_range_clamp=True, enable_parallel_detect=True
        )

    # Fit every part to the input audio length exactly. Lead is
    # quantised by frame-bucket rounding; SATB are stochastically
    # generated and then mutated by coalesce/grid-align — neither
    # route guarantees the right duration. Running fit last makes the
    # output MIDI's nominal length equal the input MP3's length when
    # played at the tempo written into the MIDI.
    lead_tokens = fit_to_length(lead_tokens, target_sixteenths)
    for voice in VOICES:
        generated[voice] = fit_to_length(generated[voice], target_sixteenths)
        logger.info("voice %s: %d tokens (fit to %d sixteenths)",
                    voice, len(generated[voice]), target_sixteenths)

    # Derive the MIDI tempo from target_sixteenths and target_duration
    # so playback of exactly ``target_sixteenths`` 16th-notes lasts
    # exactly ``target_duration_sec``. Differs from the tokenizer tempo
    # ``tempo_used`` only by the sub-1/BPM drift introduced when
    # rounding the sixteenth count — typically < 0.02 BPM, well below
    # any perceptual threshold.
    if target_duration_sec > 0 and target_sixteenths > 0:
        midi_tempo = 60.0 * target_sixteenths / (4.0 * target_duration_sec)
    else:
        midi_tempo = tempo_used
    logger.info(
        "midi tempo %.6f bpm (tokenizer tempo was %.3f; delta %.4f)",
        midi_tempo, tempo_used, midi_tempo - tempo_used,
    )

    if out_path is None:
        out_path = audio_path.with_name(f"{audio_path.stem}_arrangement.mid")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # mido-based writer: exact ticks, exact microseconds-per-quarter,
    # no measure padding (which music21 did) and no tempo rounding
    # surprise (which pretty_midi did).
    _write_exact_midi(
        lead_tokens,
        generated,
        tempo_bpm=midi_tempo,
        target_sixteenths=target_sixteenths,
        out_path=out_path,
    )
    logger.info("wrote %s", out_path)
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arrange an audio file as a four-voice SATB MIDI."
    )
    p.add_argument("--input", type=Path, required=True, help="Path to the audio file.")
    p.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to a trained .pt checkpoint."
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output .mid path. Defaults to <input>_arrangement.mid alongside the input.",
    )
    p.add_argument(
        "--model-class", choices=("hybrid", "baseline"), default="hybrid",
    )
    p.add_argument(
        "--no-voice-leading", dest="voice_leading", action="store_false", default=True,
        help="Disable the voice-leading post-process (for ablation comparison).",
    )
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--duration-temperature", type=float, default=1.1)
    p.add_argument("--bass-temperature", type=float, default=0.3,
                   help="Pitch-sampling temperature for bass only. Default 0.3 "
                        "yields stable root-note bass lines; raise toward the "
                        "global --temperature if you want a walking bass.")
    p.add_argument("--bass-duration-temperature", type=float, default=0.7,
                   help="Duration temperature for bass only. Default 0.7 "
                        "biases bass toward long held notes.")
    p.add_argument("--satb-upper-temperature", type=float, default=0.35,
                   help="Pitch-sampling temperature shared by S/A/T. Default "
                        "0.35 — colder than the legacy --temperature to keep "
                        "upper voices from generating a new note every 16th. "
                        "See docs/issues/004.")
    p.add_argument("--satb-upper-duration-temperature", type=float, default=0.8,
                   help="Duration temperature shared by S/A/T. Default 0.8 "
                        "concentrates durations above the 16th-note bucket "
                        "while keeping more variety than bass's 0.7.")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--tempo-bpm", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    out_path = run_pipeline(
        audio_path=args.input,
        model_checkpoint=args.checkpoint,
        out_path=args.out,
        voice_leading=args.voice_leading,
        model_class=args.model_class,
        temperature=args.temperature,
        duration_temperature=args.duration_temperature,
        bass_temperature=args.bass_temperature,
        bass_duration_temperature=args.bass_duration_temperature,
        satb_upper_temperature=args.satb_upper_temperature,
        satb_upper_duration_temperature=args.satb_upper_duration_temperature,
        top_k=args.top_k,
        max_len=args.max_len,
        tempo_bpm=args.tempo_bpm,
        device=args.device,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
