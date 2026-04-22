"""One-time helper: push ``phase_b_final.pt`` to a Hugging Face Hub
model repo so the deployed Streamlit Space can pull it at runtime.

Why this exists
---------------

The trained checkpoint is 90 MB, which is over GitHub's single-file
ceiling. We can't ship it in the repo, so the app downloads it from
HF Hub on first launch via ``huggingface_hub.hf_hub_download``. This
script is the one-time upload step that seeds that Hub repo.

Usage
-----

1. Install deps: ``pip install huggingface_hub`` (already in
   ``requirements.txt``).
2. Log in: ``huggingface-cli login`` — accept a write-scope token
   from https://huggingface.co/settings/tokens.
3. Run this script, optionally overriding defaults:

   ::

       python scripts/upload_checkpoint_to_hf.py \
           --checkpoint checkpoints/phase_b/phase_b_final.pt \
           --repo-id <your-hf-username>/aca-adapt-model

4. Set ``ACA_ADAPT_HF_REPO_ID`` in the Streamlit Space's
   **Variables and secrets** to the same ``<user>/aca-adapt-model``
   string. The app will ``hf_hub_download`` the checkpoint on first
   launch and cache it under ``~/.cache/aca-adapt/``.

If the target repo doesn't exist, this script creates it (public by
default; pass ``--private`` to make it private — the app still
works but you'll need to supply ``HF_TOKEN`` as a Space secret).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

_DEFAULT_CHECKPOINT = Path("checkpoints/phase_b/phase_b_final.pt")
_DEFAULT_REMOTE_NAME = "phase_b_final.pt"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        help=f"Local path to the .pt checkpoint (default: {_DEFAULT_CHECKPOINT}).",
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="Target repo on HF Hub in the form '<username>/<reponame>'. "
             "The repo will be created if it doesn't exist.",
    )
    p.add_argument(
        "--remote-filename",
        default=_DEFAULT_REMOTE_NAME,
        help=f"Filename to store on HF Hub (default: {_DEFAULT_REMOTE_NAME}). "
             "Must match ACA_ADAPT_HF_FILENAME on the Space side.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private. Requires HF_TOKEN on the Space.",
    )
    p.add_argument(
        "--commit-message",
        default="upload phase_b checkpoint",
        help="Commit message recorded on the HF Hub repo.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        print(f"checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    size_mb = ckpt.stat().st_size / (1024 * 1024)
    print(f"uploading {ckpt.name} ({size_mb:.1f} MB) to {args.repo_id}")

    api = HfApi()
    # Idempotent: create_repo with exist_ok=True lets the script
    # double as the "initial setup" and the "retrain and re-upload"
    # flow without branching on whether the repo exists.
    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo=args.remote_filename,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(
        f"done. Set the Streamlit Space's ACA_ADAPT_HF_REPO_ID secret to "
        f"'{args.repo_id}' (and ACA_ADAPT_HF_FILENAME to "
        f"'{args.remote_filename}' if you overrode the default)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
