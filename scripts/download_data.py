"""Fetch training data for Aca-Adapt.

Downloads jaCappella (MusicXML only, ~tens of MB — skips the 4 GB audio)
from HuggingFace and verifies that JSB Chorales are reachable through
music21.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --skip-jacappella  # JSB check only
    python scripts/download_data.py --token HF_xxx     # explicit token

Requires:
    - HuggingFace account that has accepted the jaCappella license
      (https://huggingface.co/datasets/jaCappella/jaCappella)
    - `huggingface-cli login` run once, OR HF_TOKEN env var set,
      OR the --token flag above.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JACAPPELLA_DIR = PROJECT_ROOT / "data" / "raw" / "jacappella"


def download_jacappella(dest: Path, token: str | None = None) -> int:
    """Download only the symbolic score files + metadata from jaCappella."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required.\n"
            "  pip install huggingface_hub"
        )

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading jaCappella MusicXML files to {dest} ...")

    try:
        snapshot_download(
            repo_id="jaCappella/jaCappella",
            repo_type="dataset",
            local_dir=str(dest),
            # Only pull scores + metadata; skip ~4 GB of audio.
            allow_patterns=["**/*.musicxml", "meta.csv", "*.txt"],
            token=token,
        )
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("401", "403", "unauthorized", "gated", "restricted")):
            sys.exit(
                "\njaCappella is a gated dataset. Before running again:\n"
                "  1. Create / sign in to a HuggingFace account.\n"
                "  2. Accept the license at:\n"
                "       https://huggingface.co/datasets/jaCappella/jaCappella\n"
                "  3. Run `huggingface-cli login` with a read token from:\n"
                "       https://huggingface.co/settings/tokens\n"
            )
        raise

    musicxml_count = len(list(dest.rglob("*.musicxml")))
    print(f"  -> {musicxml_count} MusicXML files on disk.")
    return musicxml_count


def verify_jsb_chorales() -> int:
    """Load JSB chorales through music21 and return how many are available."""
    try:
        from music21 import corpus
    except ImportError:
        sys.exit(
            "music21 is required.\n"
            "  pip install music21"
        )

    try:
        chorales = list(corpus.chorales.Iterator())
    except Exception as e:
        sys.exit(f"Failed to load JSB chorales via music21: {e}")

    if not chorales:
        sys.exit(
            "music21 returned zero chorales. The core corpus may not be "
            "installed. Try:\n"
            "  python -c \"from music21 import configure; configure.run()\""
        )

    print(f"JSB Chorales available via music21: {len(chorales)} pieces.")
    return len(chorales)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace read token (falls back to HF_TOKEN env var or cached login).",
    )
    parser.add_argument(
        "--skip-jacappella",
        action="store_true",
        help="Only verify JSB chorales (skip jaCappella download).",
    )
    parser.add_argument(
        "--skip-jsb",
        action="store_true",
        help="Only download jaCappella (skip JSB verification).",
    )
    args = parser.parse_args()

    if not args.skip_jacappella:
        download_jacappella(JACAPPELLA_DIR, token=args.token)

    if not args.skip_jsb:
        verify_jsb_chorales()

    print("\nAll set. Next: run notebooks/01_data_exploration.py.")


if __name__ == "__main__":
    main()
