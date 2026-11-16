#!/usr/bin/env python3
"""Download the InternVLA-N1 DualVLN checkpoint.

Fetches, from the Hugging Face Hub:
  - InternRobotics/InternVLA-N1-DualVLN  -> pretrained/InternVLA-N1-DualVLN/

Usage:
    python download_checkpoints.py [--output-dir pretrained]
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "InternRobotics/InternVLA-N1-DualVLN"


def download_checkpoint(output_dir: Path) -> Path:
    dest = Path(output_dir) / REPO_ID.split("/")[-1]
    print(f"Downloading {REPO_ID} -> {dest}")
    # local_dir avoids cache symlinks, which fail on Windows without privileges (WinError 1314).
    snapshot_download(repo_id=REPO_ID, local_dir=dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "pretrained"),
        help="Directory to download the checkpoint into (default: %(default)s)",
    )
    args = parser.parse_args()
    dest = download_checkpoint(Path(args.output_dir))
    print("\nDone. Contents of", dest)
    for entry in sorted(dest.iterdir()):
        print(" ", entry.name)


if __name__ == "__main__":
    main()
