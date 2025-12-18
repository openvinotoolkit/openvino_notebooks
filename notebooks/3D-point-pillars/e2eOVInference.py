"""
End-to-End OpenVINO Inference for PointPillars
Uses custom extensions for voxelization and post-processing
"""

import json
import numpy as np
from openvino import Core, Tensor


class E2EOVInference:
    """
    End-to-End OpenVINO inference engine for PointPillars
    Uses custom extensions for voxelization and post-processing
    """

    def __init__(self, config_path='pretrained/pointpillars_full_config.json', device='CPU'):
        """
        Initialize OpenVINO inference engine

        Args:
            config_path: Path to configuration JSON file
            device: OpenVINO device ('CPU', 'GPU', etc.)
        """
        self.device = device.upper()
        print(f"Initializing E2E OpenVINO backend on {self.device}...")

        # Load models and extension
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.core = Core()
        self.core.add_extension(config['extension_lib'])

        # Load and compile models
        voxel_model = self.core.read_model(config['voxel_model'])
        self.compiled_voxel = self.core.compile_model(voxel_model, 'CPU')

        nn_model = self.core.read_model(config['nn_model'])
        self.compiled_nn = self.core.compile_model(nn_model, self.device)

        postproc_model = self.core.read_model(config['postproc_model'])
        self.compiled_postproc = self.core.compile_model(postproc_model, 'CPU')

        print(f"✓ OpenVINO models loaded:")
        print("  - Voxelization (custom extension) in CPU")
        print(f"  - Neural Network (OpenVINO IR) in {self.device}")
        print("  - Post-processing (custom extension) in CPU")

    def _voxelize(self, points):
        """Voxelization using OpenVINO custom extension"""
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(-1, 4)

        infer_request = self.compiled_voxel.create_infer_request()
        infer_request.set_input_tensor(0, Tensor(points))
        infer_request.infer()

        pillars = infer_request.get_output_tensor(0).data
        coors = infer_request.get_output_tensor(1).data
        npoints = infer_request.get_output_tensor(2).data

        return pillars, coors, npoints

    def _run_nn(self, pillars, coors, npoints):
        """Neural network inference"""
        infer_request = self.compiled_nn.create_infer_request()
        infer_request.infer({
            'pillars': pillars.astype(np.float32),
            'coors': coors.astype(np.int32),
            'npoints': npoints.astype(np.int32)
        })

        cls_preds = infer_request.get_output_tensor(0).data
        box_preds = infer_request.get_output_tensor(1).data
        dir_cls_preds = infer_request.get_output_tensor(2).data

        return cls_preds, box_preds, dir_cls_preds

    def _postprocess(self, cls_preds, box_preds, dir_cls_preds):
        """Post-processing using OpenVINO custom extension"""
        # Remove batch dimension
        cls_preds = np.squeeze(cls_preds, axis=0)
        box_preds = np.squeeze(box_preds, axis=0)
        dir_cls_preds = np.squeeze(dir_cls_preds, axis=0)

        infer_request = self.compiled_postproc.create_infer_request()
        infer_request.set_input_tensor(0, Tensor(cls_preds))
        infer_request.set_input_tensor(1, Tensor(box_preds))
        infer_request.set_input_tensor(2, Tensor(dir_cls_preds))
        infer_request.infer()

        bboxes = infer_request.get_output_tensor(0).data
        labels = infer_request.get_output_tensor(1).data
        scores = infer_request.get_output_tensor(2).data

        return bboxes, labels, scores

    def infer(self, points):
        """
        Run inference on point cloud

        Args:
            points: numpy array of shape [N, 4] (x, y, z, intensity)

        Returns:
            dict with keys: 'lidar_bboxes', 'labels', 'scores'
        """
        # Step 1: Voxelization
        pillars, coors, npoints = self._voxelize(points)

        # Step 2: Neural network
        cls_preds, box_preds, dir_cls_preds = self._run_nn(pillars, coors, npoints)

        # Step 3: Post-processing
        bboxes, labels, scores = self._postprocess(cls_preds, box_preds, dir_cls_preds)

        return {
            'lidar_bboxes': bboxes,
            'labels': labels,
            'scores': scores
        }


if __name__ == "__main__":
    import argparse

    def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
        """Filter points within specified range"""
        flag_x_low = pts[:, 0] > point_range[0]
        flag_y_low = pts[:, 1] > point_range[1]
        flag_z_low = pts[:, 2] > point_range[2]
        flag_x_high = pts[:, 0] < point_range[3]
        flag_y_high = pts[:, 1] < point_range[4]
        flag_z_high = pts[:, 2] < point_range[5]
        keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
        pts = pts[keep_mask]
        return pts

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

    # Parse arguments
    parser = argparse.ArgumentParser(description='PointPillars E2E OpenVINO Inference')
    parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU'],
                        help='OpenVINO device')
    parser.add_argument('--config', default='PointPillars/pretrained/pointpillars_ov_config.json',
                        help='Path to config JSON')
    parser.add_argument('--pc_path', default='PointPillars/pointpillars/dataset/demo_data/val/000134.bin',
                        help='Path to point cloud file')
    args = parser.parse_args()

    # Initialize engine
    engine = E2EOVInference(config_path=args.config, device=args.device)

    # Load and filter point cloud
    pc = np.fromfile(args.pc_path, dtype=np.float32).reshape(-1, 4)
    pc = point_range_filter(pc)

    print(f"\n✓ Point cloud loaded: {len(pc)} points (after ROI filtering)")

    # Run inference
    result = engine.infer(pc)

    # Filter results to valid range
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)
    result_filter = keep_bbox_from_lidar_range(result, pcd_limit_range)

    # Print results
    lidar_bboxes = result_filter['lidar_bboxes']
    labels = result_filter['labels']
    scores = result_filter['scores']

    print(f"\nDetection Results:")
    print(f"Detected {len(lidar_bboxes)} objects:")
    for i, (bbox, label, score) in enumerate(zip(lidar_bboxes, labels, scores)):
        print(f"  Object {i}: class={label}, score={score:.3f}")
