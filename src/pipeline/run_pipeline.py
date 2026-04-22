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
import logging
from pathlib import Path

import torch
from music21 import stream

from src.data.tokenizer import decode_part
from src.eval.evaluate import _build_model, _load_hparams_from_sources, _resolve_device
from src.pipeline.audio_to_midi import extract_lead_tokens
from src.pipeline.decode import VOICES, generate_voice_tokens
from src.postprocess.voice_leading import apply_voice_leading

logger = logging.getLogger(__name__)


def _assemble_score(
    lead_tokens: list[int], generated: dict[str, list[int]]
) -> stream.Score:
    """Stack Lead + S/A/T/B into one Score with Lead on the top staff.

    Split out so tests can assert on the part structure in-memory —
    music21's MIDI writer does not preserve ``partName`` on round-trip,
    so file-level assertions cannot check staff order.
    """
    score = stream.Score()
    lead_part = decode_part(lead_tokens)
    lead_part.partName = "LEAD"
    score.insert(0, lead_part)
    for voice in VOICES:
        part = decode_part(generated[voice])
        part.partName = voice.upper()
        score.insert(0, part)
    return score


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
    # Bass-specific sampling overrides. The global ``temperature`` and
    # ``duration_temperature`` defaults were tuned to stop S/A/T from
    # collapsing onto the modal quarter-note bucket — the exact behaviour
    # *bass* should have. Pop bass conventions want long held notes and
    # stable pitches, so we drop both temperatures on bass only. See
    # ``docs/issues/002-bass-voice-too-short-and-jumpy.md``.
    bass_temperature: float = 0.3,
    bass_duration_temperature: float = 0.7,
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

    lead_tokens = extract_lead_tokens(audio_path, tempo_bpm=tempo_bpm, device=device)
    if len(lead_tokens) < 3:
        raise RuntimeError(
            f"extracted only {len(lead_tokens)} tokens from {audio_path.name}; "
            "audio may be silent or too short for pitch tracking"
        )
    logger.info("lead tokens: %d", len(lead_tokens))

    lead_tensor = torch.tensor([lead_tokens], dtype=torch.long, device=dev)

    model = _load_model(checkpoint_path, model_class, dev)

    generated: dict[str, list[int]] = {}
    for voice in VOICES:
        # Bass gets stricter temperatures so it reads as a pop bass line
        # (long held root notes) instead of mimicking the upper voices.
        v_temp = bass_temperature if voice == "b" else temperature
        v_dur_temp = bass_duration_temperature if voice == "b" else duration_temperature
        generated[voice] = generate_voice_tokens(
            model, lead_tensor, voice, max_len, dev,
            temperature=v_temp,
            duration_temperature=v_dur_temp,
            top_k=top_k,
        )
        logger.info("voice %s: %d tokens", voice, len(generated[voice]))

    if voice_leading:
        generated = apply_voice_leading(
            generated, enable_range_clamp=True, enable_parallel_detect=True
        )

    score = _assemble_score(lead_tokens, generated)

    if out_path is None:
        out_path = audio_path.with_name(f"{audio_path.stem}_arrangement.mid")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(out_path))
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
        top_k=args.top_k,
        max_len=args.max_len,
        tempo_bpm=args.tempo_bpm,
        device=args.device,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
