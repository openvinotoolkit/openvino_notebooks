#!/usr/bin/env python3
"""Convert KITTI split pointclouds referenced in kitti_infos_{split}.pkl to binary PCD.

Usage:
  python scripts/convert_kitti_split_to_pcd.py --data-root /path/to/dataset --split val

Example:
  python pointpillars/utils/convert_kitti_split_to_pcd.py --data-root Datasets --split val  --overwrite --verify

The script reads Datasets/kitti_infos_{split}.pkl and converts each referenced
velodyne .bin file (x,y,z,intensity float32) to a binary .pcd with ALL 4 fields
preserved (x, y, z, intensity). The PCD format is PCL-compatible.

To read the PCD files back as Nx4 arrays, use read_kitti_pcd() from this module.
"""
import argparse
import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pointpillars.dataset.kitti_open3d.utils import (
    convert_bin_to_pcd,
    compare_bin_pcd_file,
)


def main():
    parser = argparse.ArgumentParser(description='Convert KITTI split to PCD')
    parser.add_argument('--data-root', default='Datasets', help='KITTI-like data root')
    parser.add_argument('--split', default='val', choices=['train','val','trainval','test'])
    parser.add_argument('--out-dir', default=None, help='Output directory for pcd files (default: <data-root>/pcd_<split> or derived from pts-prefix)')
    parser.add_argument('--pts-prefix', default='velodyne_reduced', help='Pointcloud folder prefix to use (mirrors Kitti default)')
    parser.add_argument('--max', type=int, default=0, help='Max number of files to convert (0=all)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing pcd files')
    parser.add_argument('--verify', action='store_true', help='After conversion, read back and compare each PCD with original .bin')
    args = parser.parse_args()

    data_root = args.data_root
    split = args.split
    pkl_path = os.path.join(data_root, f'kitti_infos_{split}.pkl')
    if not os.path.exists(pkl_path):
        raise SystemExit(f'kitti infos not found: {pkl_path}')

    with open(pkl_path, 'rb') as f:
        infos = pickle.load(f)

    pts_prefix = args.pts_prefix

    # default out_dir: if user passed --out-dir use it, otherwise create a pcd folder
    # next to the source pts folder. For KITTI-like paths 'training/velodyne/000001.bin'
    # and pts_prefix='velodyne_reduced' we'll create 'Datasets/training/velodyne_reduced_pcd'
    if args.out_dir:
        out_dir = args.out_dir
    else:
        # derive parent folder from the first entry and create sibling <pts_prefix>_pcd
        sample_key = sorted(infos.keys())[0]
        sample_vel_path = infos[sample_key].get('velodyne_path')
        parent = os.path.dirname(os.path.join(data_root, sample_vel_path))
        out_dir = os.path.join(os.path.dirname(parent), f'{pts_prefix}_pcd')

    os.makedirs(out_dir, exist_ok=True)

    print(f'Found {len(infos)} entries in {pkl_path}')

    converted = 0
    missing = 0
    verify_failed = 0
    processed = 0  # counts files that would be converted (also used by --max)
    for i, key in enumerate(sorted(infos.keys())):
        info = infos[key]
        vel_path = info.get('velodyne_path')
        if vel_path is None:
            print(f'[{key}] No velodyne_path, skipping')
            continue

        # absolute path: replace 'velodyne' component with pts_prefix (mirrors Kitti behavior)
        vel_path_resolved = vel_path.replace('velodyne', pts_prefix)
        bin_path = os.path.join(data_root, vel_path_resolved)

        base = os.path.splitext(os.path.basename(vel_path))[0]
        pcd_path = os.path.join(out_dir, f'{base}.pcd')

        if not os.path.exists(bin_path):
            print(f'[{key}] MISSING: {bin_path}')
            missing += 1
            continue

        if os.path.exists(pcd_path) and not args.overwrite:
            print(f'[{key}] SKIP (exists): {pcd_path}')
            continue

        # convert the file
        processed += 1
        print(f'[{key}] {bin_path} -> {pcd_path}')
        try:
            convert_bin_to_pcd(bin_path, pcd_path)
            converted += 1
            # Optional verification per file
            if args.verify:
                ok = compare_bin_pcd_file(bin_path, pcd_path)
                if not ok:
                    print(f'  VERIFY FAILED: {bin_path} -> {pcd_path}')
                    verify_failed += 1
        except Exception as e:
            print(f'  ERROR converting {bin_path}: {e}')

        # stop when we've processed enough files (applies to --max)
        if args.max > 0 and processed >= args.max:
            break

    print(f'Done. Converted: {converted}, Missing: {missing}, Verify failed: {verify_failed}, Output dir: {out_dir}')


if __name__ == '__main__':
    main()
