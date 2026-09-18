#!/usr/bin/env python3
"""Download every pre-trained checkpoint the export/inference pipeline needs into `pretrained/`.

Fetches, from the Hugging Face Hub:
  - Qwen/Qwen3-VL-2B-Instruct        -> pretrained/Qwen3-VL-2B-Instruct/
  - ginwind/VLA-JEPA (LIBERO/*)      -> pretrained/LIBERO/

Note: the V-JEPA2 encoder (facebook/vjepa2-vitl-fpc64-256) and the pre-fine-tuning
"Pretrain" checkpoint are only used by the original training/baseline code
(`run_baseline.py`), not by `export.py` / `run_inference_standalone.py`, so they are
not downloaded here.

Usage:
    python download_checkpoints.py [--output-dir pretrained] [--only qwen,libero]
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

VLA_JEPA_REPO = "ginwind/VLA-JEPA"
QWEN_REPO = "Qwen/Qwen3-VL-2B-Instruct"

ALL_TARGETS = ["qwen", "libero"]


def download_qwen(output_dir: Path) -> None:
    dest = output_dir / "Qwen3-VL-2B-Instruct"
    print(f"Downloading {QWEN_REPO} -> {dest}")
    snapshot_download(repo_id=QWEN_REPO, local_dir=dest)


def download_libero(output_dir: Path) -> None:
    dest = output_dir / "LIBERO"
    print(f"Downloading {VLA_JEPA_REPO}/LIBERO -> {dest}")
    with_prefix = snapshot_download(
        repo_id=VLA_JEPA_REPO,
        allow_patterns=["LIBERO/*", "LIBERO/**/*"],
    )
    src = Path(with_prefix) / "LIBERO"
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(item, target)


DOWNLOADERS = {
    "qwen": download_qwen,
    "libero": download_libero,
}


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "pretrained"),
        help="Directory to download checkpoints into (default: %(default)s)",
    )
    parser.add_argument(
        "--only",
        default=",".join(ALL_TARGETS),
        help=f"Comma-separated subset of {ALL_TARGETS} to download (default: all)",
    )
    args = parser.parse_args()

    targets = [t.strip() for t in args.only.split(",") if t.strip()]
    unknown = set(targets) - set(ALL_TARGETS)
    if unknown:
        parser.error(f"Unknown target(s): {sorted(unknown)}. Choose from {ALL_TARGETS}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        DOWNLOADERS[target](output_dir)

    print("\nDone. Contents of", output_dir)
    for entry in sorted(output_dir.iterdir()):
        print(" ", entry.name)


if __name__ == "__main__":
    main()
