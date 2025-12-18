"""
Test PointPillars OpenVINO model with E2E inference
Usage: python test_exported_model.py --device cpu --pc_path /workspace/pointpillars/dataset/demo_data/test/000002.bin
"""

import argparse
import numpy as np
import os
import sys
from e2eOVInference import E2EOVInference



def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    """Filter points within specified range"""
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    return pts[keep_mask]


def keep_bbox_from_lidar_range(result, pcd_limit_range):
    """Filter bboxes within specified range"""
    lidar_bboxes = result['lidar_bboxes']
    labels = result['labels']
    scores = result['scores']

    flag1 = lidar_bboxes[:, :3] > pcd_limit_range[:3][None, :]
    flag2 = lidar_bboxes[:, :3] < pcd_limit_range[3:][None, :]
    keep_flag = np.all(flag1, axis=-1) & np.all(flag2, axis=-1)

    return {
        'lidar_bboxes': lidar_bboxes[keep_flag],
        'labels': labels[keep_flag],
        'scores': scores[keep_flag]
    }

def get_pc_from_bin(pc_path):
    """Load point cloud from .bin file"""
    pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    pc = point_range_filter(pc)

    print(f"\n✓ Point cloud loaded: {len(pc)} points (after ROI filtering), shape: {pc.shape}")
    return pc

def get_pc_from_pcd(pc_path):
    REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PointPillars")
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from pointpillars.dataset.kitti_open3d.utils import read_kitti_pcd

    pts = read_kitti_pcd(pc_path)
    pc = point_range_filter(pts)
    print(f"\n✓ Point cloud loaded from PCD: {len(pc)} points (after ROI filtering), shape: {pc.shape}")
    return pc

def main(args):
    """Main test function"""

    # Initialize E2E OpenVINO inference engine
    print(f"Initializing PointPillars OpenVINO inference...")
    engine = E2EOVInference(
        config_path=args.config,
        device=args.device.upper()
    )

    # Load test point cloud
    if not os.path.exists(args.pc_path):
        raise FileNotFoundError(f"Point cloud not found: {args.pc_path}")

    if args.pc_path.endswith('.bin'):
        pc = get_pc_from_bin(args.pc_path)
    elif args.pc_path.endswith('.pcd'):
        pc = get_pc_from_pcd(args.pc_path)

    # Run inference
    result = engine.infer(pc)

    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    # Filter results to valid range
    result_filter = keep_bbox_from_lidar_range(result, pcd_limit_range)

    # Print results
    lidar_bboxes = result_filter['lidar_bboxes']
    labels = result_filter['labels']
    scores = result_filter['scores']

    print(f"\nDetection Results:")
    print(f"Detected {len(lidar_bboxes)} objects:")
    for i, (bbox, label, score) in enumerate(zip(lidar_bboxes, labels, scores)):
        print(f"  Object {i}: class={label}, score={score:.3f}")


if __name__ == "__main__":
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='PointPillars OpenVINO E2E Inference Test')
    parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU'],
                        help='OpenVINO device')
    parser.add_argument('--config', default=f'{current_file_path}/pretrained/pointpillars_ov_config.json',
                        help='Path to configuration JSON file')
    parser.add_argument('--pc_path', default=f'{current_file_path}/pointpillars/dataset/demo_data/test/000002.bin',
                        help='Path to point cloud file')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("PointPillars OpenVINO E2E Inference - Test")
    print("="*60)
    print("\nConfiguration:")
    print(f"  device: {args.device}")
    print(f"  config: {args.config}")
    print(f"  pc_path: {args.pc_path}")
    print("="*60 + "\n")

    main(args)
