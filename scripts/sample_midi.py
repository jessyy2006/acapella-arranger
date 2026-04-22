"""Existential-check MIDI sampler.

Loads a trained hybrid checkpoint, pulls one lead melody from a processed
split, runs greedy autoregressive decoding for all four voices, and
writes MIDI files (one with voice-leading post-process, one without) so
a human can open them in MuseScore / GarageBand and actually listen.

Usage:
    python scripts/sample_midi.py \\
        --checkpoint checkpoints/phase_b/phase_b_final.pt \\
        --split data/processed/test_jsb.pt \\
        --example 0 \\
        --out-dir outputs

The first example in ``test_jsb.pt`` is a JSB chorale excerpt; use
``test_jacappella.pt`` instead to sample against a pop-vocal lead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from music21 import stream

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_dataset
from src.data.tokenizer import decode_part
from src.data.vocab import PAD, SOS
from src.eval.evaluate import _build_model, _load_hparams_from_sources, _resolve_device
from src.pipeline.decode import VOICES as _VOICES, generate_voice_tokens
from src.postprocess.voice_leading import apply_voice_leading

logger = logging.getLogger("sample_midi")


def _assemble_score(tokens_by_voice: dict[str, list[int]]) -> stream.Score:
    """Stack four per-voice Parts into one Score so MuseScore opens it sanely."""
    score = stream.Score()
    for voice in _VOICES:
        part = decode_part(tokens_by_voice[voice])
        part.partName = voice.upper()
        score.insert(0, part)
    return score


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Greedy-decode a trained checkpoint into a MIDI.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--example", type=int, default=0, help="Index of the example in --split")
    p.add_argument("--model-class", choices=("hybrid", "baseline"), default="hybrid")
    p.add_argument("--max-len", type=int, default=256)
    # Defaults locked in after human listening tests across v1-v4. Greedy
    # (T=0) produces degenerate A/T/B; T>0.6 introduces audible dissonance
    # because the four voices generate independently (no cross-voice
    # attention in the current hybrid architecture). T=0.5 + top-k 10 is
    # the empirical sweet spot; duration-T 1.1 keeps note lengths varied
    # without collapsing on the quarter-note bucket.
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Pitch-position sampling temperature. 0.0 = greedy.")
    p.add_argument("--duration-temperature", type=float, default=1.1,
                   help="Temperature override for duration-token positions.")
    # Bass-only overrides — the global temperatures are tuned for S/A/T,
    # which need variety; bass needs the opposite (long held root notes).
    # See docs/issues/002-bass-voice-too-short-and-jumpy.md.
    p.add_argument("--bass-temperature", type=float, default=0.3,
                   help="Pitch temperature for bass only (default: stable root notes).")
    p.add_argument("--bass-duration-temperature", type=float, default=0.7,
                   help="Duration temperature for bass only (default: long held notes).")
    p.add_argument("--top-k", type=int, default=10,
                   help="Restrict sampling to top-k tokens; 0 or None disables.")
    p.add_argument("--suffix", type=str, default="",
                   help="Appended to output filenames, e.g. '_t08' to distinguish runs.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    device = _resolve_device(args.device)
    logger.info("device=%s", device)

    # Reuse the eval harness's hparam-resolution + model build, so the
    # state_dict loads against the correct architecture.
    state = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    hparams = _load_hparams_from_sources(args.checkpoint, state, args.model_class)
    if not hparams:
        raise ValueError(f"could not resolve hparams for {args.checkpoint}")
    logger.info("hparams=%s", hparams)

    model = _build_model(args.model_class, hparams).to(device).eval()
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    dataset = load_dataset(args.split)
    if args.example >= len(dataset):
        raise IndexError(f"--example {args.example} out of range ({len(dataset)} examples)")
    example = dataset[args.example]
    lead = example["lead"].unsqueeze(0).to(device)  # (1, L_lead)
    logger.info("lead length=%d tokens", lead.shape[1])

    generated: dict[str, list[int]] = {}
    for voice in _VOICES:
        # Bass gets stricter sampling than S/A/T — it's a single voice
        # with different aesthetic expectations (hold the root, don't
        # dance around). See issue 002.
        v_temp = args.bass_temperature if voice == "b" else args.temperature
        v_dur_temp = (
            args.bass_duration_temperature if voice == "b" else args.duration_temperature
        )
        generated[voice] = generate_voice_tokens(
            model, lead, voice, args.max_len, device,
            temperature=v_temp,
            duration_temperature=v_dur_temp,
            top_k=args.top_k,
        )
        logger.info("voice %s: generated %d tokens", voice, len(generated[voice]))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # VL-off: assemble and write raw.
    vl_off_score = _assemble_score(generated)
    vl_off_path = args.out_dir / f"sample_{args.checkpoint.stem}{args.suffix}_vl_off.mid"
    vl_off_score.write("midi", fp=str(vl_off_path))
    logger.info("wrote %s", vl_off_path)

    # VL-on: apply post-process, then assemble.
    generated_vl_on = apply_voice_leading(generated, enable_range_clamp=True, enable_parallel_detect=True)
    vl_on_score = _assemble_score(generated_vl_on)
    vl_on_path = args.out_dir / f"sample_{args.checkpoint.stem}{args.suffix}_vl_on.mid"
    vl_on_score.write("midi", fp=str(vl_on_path))
    logger.info("wrote %s", vl_on_path)

    # Also save the lead as its own MIDI so the listener can compare.
    lead_tokens = [int(t) for t in lead[0].cpu().tolist() if t != PAD]
    lead_part = decode_part(lead_tokens)
    lead_part.partName = "LEAD"
    lead_score = stream.Score()
    lead_score.insert(0, lead_part)
    lead_path = args.out_dir / f"sample_{args.checkpoint.stem}{args.suffix}_lead.mid"
    lead_score.write("midi", fp=str(lead_path))
    logger.info("wrote %s", lead_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
