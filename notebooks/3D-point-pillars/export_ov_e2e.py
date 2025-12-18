"""
Export PointPillars model to OpenVINO IR with custom voxelization op configuration
Usage: python export_ov_e2e.py --checkpoint pretrained/epoch_160.pth --output pretrained/pointpillars_full
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import openvino as ov

from pointpillars.model import PointPillars


class NeuralNetworkPortion(nn.Module):
    """Neural network portion: PillarEncoder + Backbone + Neck + Head"""

    def __init__(self, model):
        super().__init__()
        self.pillar_encoder = model.pillar_encoder
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.head

    def forward(self, pillars, coors, npoints):
        pillar_features = self.pillar_encoder(pillars, coors, npoints)
        xs = self.backbone(pillar_features)
        x = self.neck(xs)
        cls_preds, box_preds, dir_cls_preds = self.head(x)
        return cls_preds, box_preds, dir_cls_preds


def create_pillar_layer_ir(voxel_params, output_path):
    """Create and save PillarLayer IR model with VoxelizationOp"""

    max_voxels = int(voxel_params['max_voxels'])
    max_points = int(voxel_params['max_num_points'])
    voxel_size_str = ",".join(map(str, voxel_params['voxel_size']))
    pc_range_str = ",".join(map(str, voxel_params['point_cloud_range']))

    xml_content = f"""<?xml version="1.0"?>
<net name="pillar_layer" version="11">
    <layers>
        <layer id="0" name="points_input" type="Parameter" version="opset1">
            <data shape="-1,4" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>-1</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="voxelization" type="VoxelizationOp" version="extension">
            <data voxel_size="{voxel_size_str}"
                  point_cloud_range="{pc_range_str}"
                  max_points_per_voxel="{max_points}"
                  max_voxels="{max_voxels}"/>
            <input>
                <port id="0">
                    <dim>-1</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>-1</dim>
                    <dim>{max_points}</dim>
                    <dim>4</dim>
                </port>
                <port id="2" precision="I32">
                    <dim>-1</dim>
                    <dim>4</dim>
                </port>
                <port id="3" precision="I32">
                    <dim>-1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="voxels_output" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>-1</dim>
                    <dim>{max_points}</dim>
                    <dim>4</dim>
                </port>
            </input>
        </layer>
        <layer id="3" name="coors_output" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>-1</dim>
                    <dim>4</dim>
                </port>
            </input>
        </layer>
        <layer id="4" name="npoints_output" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>-1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="2" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="3" to-layer="4" to-port="0"/>
    </edges>
</net>
"""

    pillar_layer_xml_path = output_path + "_pillar_layer.xml"
    with open(pillar_layer_xml_path, 'w') as f:
        f.write(xml_content)

    print(f"✓ PillarLayer IR saved: {pillar_layer_xml_path}")
    return pillar_layer_xml_path


def create_postprocessing_ir(postproc_params, output_path):
    """
    Create post-processing IR using custom PostProcessingOp extension.

    This creates an OpenVINO IR XML that wraps:
    1. Anchor generation (dynamic based on feature map size)
    2. BBox decoding (anchors2bboxes transformation)
    3. Per-class NMS with rotated IoU
    4. Direction classification

    The PostProcessingOp is implemented in ov_extensions/postprocessing_op.cpp
    """

    print("Creating post-processing IR with PostProcessingOp extension...")

    # Get model configuration
    ranges = postproc_params['anchors_ranges']
    sizes = postproc_params['anchors_sizes']
    rotations = postproc_params['anchors_rotations']

    # Get post-processing parameters
    nclasses = postproc_params['nclasses']
    nms_pre = postproc_params['nms_pre']
    score_thr = postproc_params['score_thr']
    nms_thr = postproc_params['nms_thr']
    max_num = postproc_params['max_num']

    # Create XML content
    xml_content = f'''<?xml version="1.0"?>
<net name="postprocessing" version="11">
    <layers>
        <layer id="0" name="bbox_cls_pred" type="Parameter" version="opset1">
            <data shape="-1,-1,-1" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>-1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="bbox_pred" type="Parameter" version="opset1">
            <data shape="-1,-1,-1" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>-1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="bbox_dir_cls_pred" type="Parameter" version="opset1">
            <data shape="-1,-1,-1" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>-1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="postprocessing" type="PostProcessingOp" version="custom_ops">
            <data ranges="{','.join(map(str, ranges))}" sizes="{','.join(map(str, sizes))}" rotations="{','.join(map(str, rotations))}" nclasses="{nclasses}" nms_pre="{nms_pre}" score_thr="{score_thr}" nms_thr="{nms_thr}" max_num="{max_num}"/>
            <input>
                <port id="0">
                    <dim>-1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                </port>
                <port id="1">
                    <dim>-1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                </port>
                <port id="2">
                    <dim>-1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>-1</dim>
                    <dim>7</dim>
                </port>
                <port id="4" precision="I64">
                    <dim>-1</dim>
                </port>
                <port id="5" precision="FP32">
                    <dim>-1</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="bboxes" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>-1</dim>
                    <dim>7</dim>
                </port>
            </input>
        </layer>
        <layer id="5" name="labels" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>-1</dim>
                </port>
            </input>
        </layer>
        <layer id="6" name="scores" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>-1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
        <edge from-layer="3" from-port="4" to-layer="5" to-port="0"/>
        <edge from-layer="3" from-port="5" to-layer="6" to-port="0"/>
    </edges>
</net>
'''

    # Write XML file
    xml_path = output_path + "_postproc.xml"
    with open(xml_path, 'w') as f:
        f.write(xml_content)

    print(f"✓ Post-processing IR saved to: {xml_path}")
    print(f"  - Inputs: bbox_cls_pred, bbox_pred, bbox_dir_cls_pred (from NN head)")
    print(f"  - Outputs: bboxes [k,7], labels [k], scores [k]")
    print(f"  - NMS pre: {nms_pre}, Score threshold: {score_thr}, NMS IoU threshold: {nms_thr}, Max detections: {max_num}")

    return xml_path


def export_full_model_to_openvino(checkpoint_path, output_path):
    """Export PointPillars to OpenVINO with custom voxelization op support"""

    print(f"Loading PyTorch checkpoint: {checkpoint_path}")
    full_model = PointPillars()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    full_model.load_state_dict(checkpoint)
    full_model.eval()

    # Export neural network portion
    print("Exporting neural network (PillarEncoder + Backbone + Neck + Head)...")
    nn_portion = NeuralNetworkPortion(full_model)

    ov_nn_model = None

    # Extract voxelization parameters
    voxel_params = {
        'voxel_size': full_model.pillar_layer.voxel_layer.voxel_size if args.voxel_size_str is None else [float(v) for v in args.voxel_size_str.split(',')],
        'point_cloud_range': full_model.pillar_layer.voxel_layer.point_cloud_range if args.pc_range_str is None else [float(v) for v in args.pc_range_str.split(',')],
        'max_num_points': full_model.pillar_layer.voxel_layer.max_num_points if args.max_points is None else int(args.max_points),
        'max_voxels': full_model.pillar_layer.voxel_layer.max_voxels[0] if args.max_voxels is None else int(args.max_voxels)
    }

    max_voxels = voxel_params['max_voxels']
    max_points = voxel_params['max_num_points']
    voxel_size = voxel_params['voxel_size']
    point_cloud_range = voxel_params['point_cloud_range']

    dummy_pillars = torch.randn(max_voxels, max_points, 4)

    vx, vy, vz = voxel_size[0], voxel_size[1], voxel_size[2]
    x_l = int((point_cloud_range[3] - point_cloud_range[0]) / vx)
    y_l = int((point_cloud_range[4] - point_cloud_range[1]) / vy)
    z_l = int((point_cloud_range[5] - point_cloud_range[2]) / vz)
    dummy_coors = torch.empty((max_voxels, 4), dtype=torch.long)
    dummy_coors[:, 0] = 0  # batch index
    if z_l > 0:
        dummy_coors[:, 1] = torch.randint(0, z_l, (max_voxels,))
    else:
        dummy_coors[:, 1] = 0
    dummy_coors[:, 2] = torch.randint(0, y_l, (max_voxels,))
    dummy_coors[:, 3] = torch.randint(0, x_l, (max_voxels,))

    dummy_npoints = torch.randint(1, max_points, (max_voxels,)).long()

    if ov_nn_model is None:
        # Convert directly from PyTorch to OpenVINO (bypasses ONNX entirely)
        # Use torch.jit.trace with check_trace=False to suppress warnings from
        # non-deterministic scatter operations in PillarEncoder
        import warnings
        try:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                traced_nn = torch.jit.trace(
                    nn_portion,
                    (dummy_pillars, dummy_coors, dummy_npoints),
                    check_trace=False,
                    strict=False
                )
                ov_nn_model = ov.convert_model(
                    traced_nn,
                    example_input=(dummy_pillars, dummy_coors, dummy_npoints),
                    input=[
                        ov.PartialShape([-1, max_points, 4]),  # pillars
                        ov.PartialShape([-1, 4]),              # coors
                        ov.PartialShape([-1]),                 # npoints
                    ]
                )
            print("✓ PyTorch model converted to OpenVINO IR directly")
        except Exception as e:
            print(f"✗ Direct conversion failed: {e}")

    # if ov_nn_model is None:
    #     # Trace directly is also throwing warnings:
    #     # traced function is producing different numeric outputs from
    #     # the original Python function on the same inputs
    #     #
    #     with torch.no_grad():
    #         traced_nn = torch.jit.trace(nn_portion, (dummy_pillars, dummy_coors, dummy_npoints), check_trace=True)
    #     ov_nn_model = ov.convert_model(traced_nn, example_input=(dummy_pillars, dummy_coors, dummy_npoints))

    if ov_nn_model is not None:
        nn_xml_path = output_path + "_nn.xml"
        ov.save_model(ov_nn_model, nn_xml_path)
        print(f"✓ Neural network IR saved: {nn_xml_path}")
    else:
        print("✗ Neural network IR conversion failed:")
        return -1

    # Create PillarLayer IR model
    print("Creating PillarLayer IR model with VoxelizationOp...")
    pillar_layer_xml_path = create_pillar_layer_ir(voxel_params, output_path)

    # Get post-processing parameters
    postproc_params = {
        'nclasses': full_model.nclasses if args.nclasses is None else int(args.nclasses),
        'nms_pre': full_model.nms_pre if args.nms_pre is None else int(args.nms_pre),
        'score_thr': full_model.score_thr if args.score_thr is None else float(args.score_thr),
        'nms_thr': full_model.nms_thr if args.nms_thr is None else float(args.nms_thr),
        'max_num': full_model.max_num if args.max_num is None else int(args.max_num),
        # anchors_ranges/sizes stored here as flat lists of floats to match create_postprocessing_ir expectations
        'anchors_ranges': (
            [v for r in full_model.anchors_generator.ranges for v in r]
            if args.anchors_ranges is None
            else [float(v) for v in args.anchors_ranges.split(',')]
        ),
        'anchors_sizes': (
            [v for s in full_model.anchors_generator.sizes for v in s]
            if args.anchors_sizes is None
            else [float(v) for v in args.anchors_sizes.split(',')]
        ),
        'anchors_rotations': (
            list(full_model.anchors_generator.rotations)
            if args.anchors_rotations is None
            else [float(v) for v in args.anchors_rotations.split(',')]
        ),
    }

    # Create Post-processing IR model
    postproc_xml_path = create_postprocessing_ir(postproc_params, output_path)

    extension_lib_path = args.extension_lib

    if not os.path.exists(extension_lib_path):
        print(f"ERROR: OpenVINO extension library not found at {extension_lib_path}")
        print("Build the extension: cd ov_extensions && bash build.sh")
        return None

    # Save configuration
    config = {
        'voxel_params': voxel_params,
        'extension_lib': extension_lib_path,
        'voxel_model': pillar_layer_xml_path,
        'nn_model': nn_xml_path,
        'postproc_model': postproc_xml_path
    }

    config_path = output_path + "_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved: {config_path}")

    print("\n" + "="*70)
    print("EXPORT COMPLETE - Full OpenVINO Pipeline")
    print("="*70)
    print("Components:")
    print(f"  1. PillarLayer IR: {pillar_layer_xml_path}")
    print(f"  2. Neural Network IR: {nn_xml_path}")
    print(f"  3. Post-processing IR: {postproc_xml_path}")
    print(f"  4. Configuration: {config_path}")
    print(f"  5. Custom Extension: {extension_lib_path}")
    print("\nModel architecture:")
    print("  - Pillar Layer (voxelization) → Custom OpenVINO Op")
    print("  - PillarEncoder → OpenVINO IR")
    print("  - Backbone → OpenVINO IR")
    print("  - Neck → OpenVINO IR")
    print("  - Head → OpenVINO IR")
    print("  - Post-processing (Anchors + BBox Decode + Rotated NMS) → Custom OpenVINO Op")
    print("\nRuntime usage:")
    print("  core = ov.Core()")
    print(f"  core.add_extension('{extension_lib_path}')")
    print(f"  pillar_layer_model = core.read_model('{pillar_layer_xml_path}')")
    print(f"  nn_model = core.read_model('{nn_xml_path}')")
    print(f"  postproc_model = core.read_model('{postproc_xml_path}')")
    print("="*70)

    return True


if __name__ == "__main__":
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Export PointPillars to OpenVINO')
    parser.add_argument('--checkpoint', type=str,
                        default=f'{current_file_path}/pretrained/epoch_160.pth',
                        help='Path to PyTorch checkpoint (.pth)')
    parser.add_argument('--output', type=str,
                        default=f'{current_file_path}/pretrained/pointpillars_ov',
                        help='Output path (without extension)')
    # Path to compiled OpenVINO extension library for custom ops
    parser.add_argument('--extension-lib', dest='extension_lib', type=str,
                        default=f'{current_file_path}/ov_extensions/build/libov_pointpillars_extensions.so',
                        help=f'Path to OpenVINO extension lib (default: {current_file_path}/ov_extensions/build/libov_pointpillars_extensions.so)')
    # Voxelization and point-cloud parameters (model-dependent)
    parser.add_argument('--max-voxels', dest='max_voxels', type=int, default=None,
                        help='Maximum number of voxels (default: 16000, from PointPillars model description)')
    parser.add_argument('--max-points', dest='max_points', type=int, default=None,
                        help='Maximum number of points per voxel (default: 32, from PointPillars model description)')
    parser.add_argument('--voxel-size-str', dest='voxel_size_str', type=str, default=None,
                        help='Voxel size as comma-separated string, default: "0.16,0.16,4", from PointPillars model description')
    parser.add_argument('--pc-range-str', dest='pc_range_str', type=str,
                        default=None,
                        help='Point cloud range as comma-separated string, "x_min,y_min,z_min,x_max,y_max,z_max", default: "0,-39.68,-3,69.12,39.68,1", from PointPillars model description')
    # Post-processing parameters (model-dependent)
    parser.add_argument('--nclasses', dest='nclasses', type=int, default=None,
                        help='Number of classes (default: 3, from PointPillars model description)')
    parser.add_argument('--nms-pre', dest='nms_pre', type=int, default=None,
                        help='Number of pre-NMS top proposals to keep (default: 100, from PointPillars model description)')
    parser.add_argument('--score-thr', dest='score_thr', type=float, default=None,
                        help='Score threshold for detections (default: 0.1, from PointPillars model description)')
    parser.add_argument('--nms-thr', dest='nms_thr', type=float, default=None,
                        help='NMS IoU threshold (default: 0.01, from PointPillars model description)')
    parser.add_argument('--max-num', dest='max_num', type=int, default=None,
                        help='Maximum number of final detections (default: 50, from PointPillars model description)')
    # Anchor configuration (comma-separated values)
    parser.add_argument('--anchors-ranges', dest='anchors_ranges', type=str,
                        default=None,
                        help='Anchors ranges as comma-separated floats (default, from PointPillars model description: "0,-39.68,-0.6,69.12,39.68,-0.6,0,-39.68,-0.6,69.12,39.68,-0.6,0,-39.68,-1.78,69.12,39.68,-1.78")')
    parser.add_argument('--anchors-sizes', dest='anchors_sizes', type=str,
                        default=None,
                        help='Anchors sizes as comma-separated floats (default, from PointPillars model description: "0.6,0.8,1.73,0.6,1.76,1.73,1.6,3.9,1.56")')
    parser.add_argument('--anchors-rotations', dest='anchors_rotations', type=str,
                        default=None,
                        help='Anchors rotations as comma-separated floats (default, from PointPillars model description: "0,1.57")')

    args = parser.parse_args()
    export_full_model_to_openvino(args.checkpoint, args.output)
