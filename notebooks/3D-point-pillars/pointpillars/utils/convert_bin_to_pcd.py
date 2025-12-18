#!/usr/bin/env python3
"""Convert a single KITTI .bin file to .pcd (saved alongside .bin) and optionally verify.

Usage:
  python scripts/convert_bin_to_pcd.py path/to/file.bin
  python scripts/convert_bin_to_pcd.py path/to/file.bin --verify
  python scripts/convert_bin_to_pcd.py path/to/file.bin --overwrite --verify

Example:
  python pointpillars/utils/convert_bin_to_pcd.py  pointpillars/dataset/demo_data/val/000134.bin --overwrite --verify
"""
import argparse
import os
import sys

# Ensure repo root is on sys.path so `pointpillars` package can be imported
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pointpillars.dataset.kitti_open3d.utils import (
    convert_bin_to_pcd,
    compare_bin_pcd_file,
)

def main():
    parser = argparse.ArgumentParser(
        description="Convert a single KITTI .bin file to .pcd (saved alongside .bin)")
    parser.add_argument("bin_path", help="Path to .bin file")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing .pcd file")
    parser.add_argument("--verify", action="store_true",
                        help="After conversion, read back and compare with original .bin")
    args = parser.parse_args()

    bin_path = args.bin_path
    if not os.path.isfile(bin_path):
        print(f"File not found: {bin_path}", file=sys.stderr)
        sys.exit(1)

    if not bin_path.lower().endswith('.bin'):
        print(f"Expected .bin file, got: {bin_path}", file=sys.stderr)
        sys.exit(1)

    # Derive output path (same location, .pcd extension)
    pcd_path = os.path.splitext(bin_path)[0] + '.pcd'

    if os.path.exists(pcd_path) and not args.overwrite:
        print(f"PCD already exists (use --overwrite to replace): {pcd_path}")
        sys.exit(0)

    # Convert
    print(f"Converting: {bin_path} -> {pcd_path}")
    try:
        convert_bin_to_pcd(bin_path, pcd_path)
        print("✓ Conversion complete")
    except Exception as e:
        print(f"✗ Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Optionally verify
    if args.verify:
        print("\nVerifying...")
        if compare_bin_pcd_file(bin_path, pcd_path):
            print("✓ Verification passed")
        else:
            print("✗ Verification failed", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
