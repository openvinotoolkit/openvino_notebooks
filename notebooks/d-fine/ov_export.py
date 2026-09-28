#!/usr/bin/env python
#
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
Export D-FINE to an OpenVINO IR.

Precisions:
    fp16     - FP16 weights, FP16 runtime.
    auto-opt - INT8 backbone (NNCF) plus FP16 for every other region.

Run them with `ov_infer.py`.

Examples
--------
    cd /workspace

    python ov_export.py --model n --precision fp16
    python ov_export.py --model n --precision auto-opt
    python ov_export.py --model l --precision auto-opt --verify
"""

import argparse
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import nncf
import openvino as ov
import torch
import torch.nn as nn
from openvino import opset13 as opset

from src.core import YAMLConfig
from two_stage_topk import rewrite_topk

MODELS = ("n", "s", "m", "l", "x")

# Backbone is quantized to INT8; all other regions are pinned to FP16.
DEC = r"__module\.model\.decoder"
ENC = r"__module\.model\.encoder"

REGION_PATTERNS = {
    "backbone": [r"__module\.model\.backbone"],
    "enc_aifi": [ENC + r"\.encoder\."],
    "enc_input_proj": [ENC + r"\.input_proj"],
    "enc_fpn": [ENC + r"\.(fpn_blocks|lateral_convs)"],
    "enc_pan": [ENC + r"\.(pan_blocks|downsample_convs)"],
    "dec_input_proj": [DEC + r"\.input_proj"],
    "dec_enc_output": [DEC + r"\.enc_output", DEC + r"\.enc_score_head", DEC + r"\.enc_bbox_head"],
    "dec_query_pos": [DEC + r"\.query_pos_head"],
    "dec_self_attn": [DEC + r"\.decoder\.layers\.\d+\.self_attn"],
    "dec_cross_attn": [DEC + r"\.decoder\.layers\.\d+\.cross_attn"],
    "dec_layer_ffn": [DEC + r"\.decoder\.layers\.\d+\.(linear|norm|gateway|activation)"],
    "dec_bbox_head": [DEC + r"\.dec_bbox_head", DEC + r"\.pre_bbox_head"],
    "dec_score_head": [DEC + r"\.dec_score_head"],
    "dec_lqe": [DEC + r"\.decoder\.lqe_layers"],
    "dec_integral": [DEC + r"\.integral"],
    "postprocessor": [r"__module\.postprocessor"],
}
REGIONS = tuple(REGION_PATTERNS)

INT8_REGIONS = ("backbone",)
F16_REGIONS = tuple(r for r in REGIONS if r not in INT8_REGIONS)

SUFFIX = {"fp16": "_fp16", "auto-opt": "_auto_opt"}

# images is NCHW; orig_sizes is one (height, width) pair per image.
IMAGE_CHANNELS = 3
SIZE_DIMS = 2

# Batch > 1 keeps the batch dimension symbolic during tracing.
TRACE_BATCH = 2

# Batch value that leaves the dimension dynamic in a PartialShape.
DYNAMIC_BATCH = -1


def input_shapes(batch, imgsz):
    """PartialShapes for (images, orig_sizes) at the given batch."""
    return {
        "images": ov.PartialShape([batch, IMAGE_CHANNELS, imgsz, imgsz]),
        "orig_sizes": ov.PartialShape([batch, SIZE_DIMS]),
    }


def name_io(ov_model):
    """Give the two inputs and three outputs their stable tensor names."""
    ov_model.inputs[0].tensor.set_names({"images"})
    ov_model.inputs[1].tensor.set_names({"orig_sizes"})
    ov_model.outputs[0].tensor.set_names({"labels"})
    ov_model.outputs[1].tensor.set_names({"boxes"})
    ov_model.outputs[2].tensor.set_names({"scores"})


def convert(model, imgsz):
    """Trace the deploy model and return the OpenVINO IR."""
    images = torch.rand(TRACE_BATCH, IMAGE_CHANNELS, imgsz, imgsz)
    sizes = torch.tensor([[imgsz, imgsz]] * TRACE_BATCH, dtype=torch.int64)
    with torch.no_grad():
        model(images, sizes)

    ov_model = ov.convert_model(
        model,
        example_input=(images, sizes),
        input=list(input_shapes(DYNAMIC_BATCH, imgsz).values()),
    )
    name_io(ov_model)

    return ov_model


def fold_f16_converts(model):
    """Collapse Convert nodes on FP32 constants into FP16 constants."""
    # Port indices: Convert (in 0 = data, out 0 = result), Constant (out 0 = value).
    CONVERT_IN = 0
    CONVERT_OUT = 0
    CONSTANT_OUT = 0

    for op in model.get_ordered_ops():
        if op.get_type_name() != "Convert" or op.get_output_element_type(CONVERT_OUT) != ov.Type.f16:
            continue

        src = op.input_value(CONVERT_IN).get_node()
        if src.get_type_name() != "Constant" or src.get_output_element_type(CONSTANT_OUT) != ov.Type.f32:
            continue

        half = opset.constant(src.data.astype(np.float16))
        half.set_friendly_name(src.get_friendly_name() + "/f16")
        for consumer in list(op.output(CONVERT_OUT).get_target_inputs()):
            consumer.replace_source_output(half.output(CONSTANT_OUT))

    model.validate_nodes_and_infer_types()


class CalibrationDataset:
    """Re-iterable source of (images, orig_sizes) numpy pairs.

    Streams the first `subset_size` preprocessed images from the val dataloader.
    """

    def __init__(self, cfg, subset_size):
        cfg.yaml_cfg.setdefault("val_dataloader", {})
        cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
        cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = 1
        self._loader = cfg.val_dataloader
        self._subset_size = subset_size

    def __len__(self):
        return self._subset_size

    def __iter__(self):
        count = 0
        for samples, targets in self._loader:
            if count >= self._subset_size:
                break

            images = np.ascontiguousarray(samples.numpy(), dtype=np.float32)
            sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            sizes = np.ascontiguousarray(sizes.numpy(), dtype=np.int64)

            yield {"images": images, "orig_sizes": sizes}

            count += 1


def build_calibration_dataset(cfg, subset_size):
    """Wrap the val subset in an nncf.Dataset (identity transform: items are inputs)."""
    source = CalibrationDataset(cfg, subset_size)

    return nncf.Dataset(source), len(source)


def region_patterns(region_names):
    """Flatten the name-regexes for a set of regions."""
    patterns = []
    for name in region_names:
        patterns.extend(REGION_PATTERNS[name])
    return patterns


def nncf_enum(nncf_module, enum_cls, value):
    """Resolve a case-insensitive NNCF enum member (TargetDevice/QuantizationPreset)."""
    try:
        return getattr(getattr(nncf_module, enum_cls), value.upper())
    except AttributeError:
        raise ValueError(f"Unsupported NNCF {enum_cls}: {value}")


def quantize_int8(ov_model, calibration_dataset, subset_size, int8_regions, target_device, preset):
    """Run NNCF INT8 PTQ over `int8_regions` only; regions elsewhere are ignored."""
    ignored_regions = tuple(r for r in REGIONS if r not in int8_regions)
    patterns = region_patterns(ignored_regions)

    # validate=False: an unmatched region pattern is a no-op.
    ignored_scope = nncf.IgnoredScope(patterns=patterns, validate=False) if patterns else None

    return nncf.quantize(
        ov_model,
        calibration_dataset,
        subset_size=subset_size,
        model_type=nncf.ModelType.TRANSFORMER,
        target_device=nncf_enum(nncf, "TargetDevice", target_device),
        preset=nncf_enum(nncf, "QuantizationPreset", preset),
        ignored_scope=ignored_scope,
    )


def region_op_names(model, region_names):
    """Friendly names of every graph op whose name matches one of the regions."""
    if not region_names:
        return set()

    regexes = [re.compile(p) for p in region_patterns(region_names)]
    names = set()

    for op in model.get_ordered_ops():
        fname = op.get_friendly_name()
        if any(rx.search(fname) for rx in regexes):
            names.add(fname)

    return names


def is_float(output):
    return output.get_element_type() in (ov.Type.f32, ov.Type.f16, ov.Type.bf16)


def wrap_f16_island(model, region_names):
    """Pin `region_names` to FP16 via explicit Convert nodes on the region boundary.

    Every f32 input of a region op is cast to f16, and every region output
    feeding outside the region is cast back to f32.
    """
    # Port index of the Convert op's single output.
    CONVERT_OUT = 0

    names = set(region_names)

    for op in list(model.get_ordered_ops()):
        if op.get_friendly_name() not in names:
            continue
        if op.get_type_name() in ("Constant", "Parameter", "Result"):
            continue

        for inp in op.inputs():
            src = inp.get_source_output()
            if src.get_element_type() == ov.Type.f32:
                inp.replace_source_output(opset.convert(src, ov.Type.f16).output(CONVERT_OUT))

        for out in op.outputs():
            if not is_float(out):
                continue

            for tgt in list(out.get_target_inputs()):
                node = tgt.get_node()
                if node.get_friendly_name() in names and node.get_type_name() != "Result":
                    continue

                tgt.replace_source_output(opset.convert(out, ov.Type.f32).output(CONVERT_OUT))

    model.validate_nodes_and_infer_types()


def build_auto_opt(ov_model, cfg, args):
    """Quantize the backbone to INT8 and pin every remaining region to FP16."""
    print("\n=== Auto-opt export (NNCF) ===")
    print(f"  device       : {args.device}")
    print(f"  INT8         : {', '.join(INT8_REGIONS)}")
    print(f"  FP16         : {', '.join(F16_REGIONS)}")

    calib, calib_size = build_calibration_dataset(cfg, args.calib_images)
    print(f"  calibration  : {calib_size} images")

    quantized = quantize_int8(ov_model, calib, calib_size, INT8_REGIONS, args.device, "mixed")
    # NNCF returns a fresh model; re-apply the tensor names.
    name_io(quantized)

    f16_names = region_op_names(quantized, F16_REGIONS)
    if f16_names:
        wrap_f16_island(quantized, f16_names)
    fold_f16_converts(quantized)

    return quantized


def save_ir(ov_model, xml_path, topk_k):
    """Write the IR with the two-stage TopK applied."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_xml = os.path.join(tmp, os.path.basename(xml_path))
        ov.save_model(ov_model, raw_xml, compress_to_fp16=True)
        rewrite_topk(raw_xml, xml_path, topk_k)


def get_real_sample(cfg, imgsz):
    """Fetch one real preprocessed image and its original size from the val dataloader.

    Returns (images[1,3,imgsz,imgsz], orig_sizes[1,2]), or None if the dataset
    is unavailable.
    """
    try:
        cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
        cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = 1
        loader = cfg.val_dataloader
        samples, targets = next(iter(loader))
    except Exception as exc:  # dataset not mounted / not configured
        print(f"  (skipping real-image sample: {exc})")
        return None

    orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    if samples.shape[-1] != imgsz or samples.shape[-2] != imgsz:
        print(f"  (val image is {tuple(samples.shape[-2:])}, expected " f"{imgsz}x{imgsz}; using it anyway)")

    return samples, orig_sizes


def match_detections(pt_boxes, pt_labels, ov_boxes, ov_labels, iou_gate=0.5):
    """Greedy 1:1 match of PT to OV detections by box IoU (xyxy, absolute)."""

    def iou(a, b):
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])

        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        inter = iw * ih

        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    ious, label_ok = [], []
    n_unmatched = 0
    used = set()
    for i in range(len(pt_boxes)):
        best_j, best_iou = -1, -1.0
        for j in range(len(ov_boxes)):
            if j in used:
                continue
            cur = iou(pt_boxes[i], ov_boxes[j])
            if cur > best_iou:
                best_iou, best_j = cur, j

        if best_j >= 0 and best_iou >= iou_gate:
            used.add(best_j)
            ious.append(best_iou)
            label_ok.append(bool(pt_labels[i] == ov_labels[best_j]))
        else:
            n_unmatched += 1

    return ious, label_ok, n_unmatched


def verify(xml_path, torch_model, cfg, args):
    """Run one val image through PyTorch and the IR and compare the detections."""
    sample = get_real_sample(cfg, args.imgsz)
    if sample is None:
        torch.manual_seed(0)
        sample = (torch.rand(1, IMAGE_CHANNELS, args.imgsz, args.imgsz), torch.tensor([[args.imgsz, args.imgsz]], dtype=torch.int64))
    images, sizes = sample

    with torch.no_grad():
        pt_labels, pt_boxes, pt_scores = torch_model(images, sizes)
    pt_labels = pt_labels.numpy()[0]
    pt_boxes = pt_boxes.numpy()[0]
    pt_scores = pt_scores.numpy()[0]

    core = ov.Core()
    compiled = core.compile_model(xml_path, args.device, {"INFERENCE_PRECISION_HINT": "f16"})
    out = compiled((images.numpy(), sizes.numpy().astype(np.int64)))
    ov_labels = out[compiled.output("labels")][0]
    ov_boxes = out[compiled.output("boxes")][0]
    ov_scores = out[compiled.output("scores")][0]

    print(f"\n=== Verification ({args.device}, f16 hint) ===")
    diff = np.abs(pt_scores - ov_scores)
    print(f"  score diff        : max={diff.max():.2e}, mean={diff.mean():.2e}")

    keep_pt = np.where(pt_scores > 0.5)[0]
    keep_ov = np.where(ov_scores > 0.5)[0]
    print(f"  detections (>0.5) : PT={len(keep_pt)}, OV={len(keep_ov)}")
    if not len(keep_pt) or not len(keep_ov):
        print("  (nothing confident enough to match)")
        return

    ious, label_ok, unmatched = match_detections(pt_boxes[keep_pt], pt_labels[keep_pt], ov_boxes[keep_ov], ov_labels[keep_ov])
    if not ious:
        print("  (no IoU>=0.5 matches)")
        return
    print(f"  matched pairs     : {len(ious)}/{len(keep_pt)} ({unmatched} unmatched)")
    print(f"  matched box IoU   : min={min(ious):.4f}, mean={sum(ious) / len(ious):.4f}")
    print(f"  label agreement   : {100.0 * sum(label_ok) / len(label_ok):.1f}%")


def build_deploy_model(config, checkpoint_path):
    """Load config + checkpoint; return (deploy_model, cfg)."""
    cfg = YAMLConfig(config, resume=checkpoint_path)

    # Do not fetch ImageNet pretrained backbone weights; the checkpoint has everything.
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
    cfg.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_sizes)

    return DeployModel().eval(), cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Export D-FINE to an OpenVINO IR (FP16 or INT8+FP16)")
    parser.add_argument(
        "--model",
        choices=MODELS,
        default=None,
        help="Model variant. Maps to configs/dfine/dfine_hgnetv2_<model>_coco.yml " "and checkpoints/dfine_<model>_coco.pth.",
    )
    parser.add_argument("-c", "--config", default=None, help="Explicit config yaml path.")
    parser.add_argument(
        "-r",
        "--checkpoint-path",
        default=None,
        help="Explicit checkpoint (.pth) path.",
    )
    parser.add_argument(
        "--precision",
        choices=tuple(SUFFIX),
        default="auto-opt",
        help="fp16 = FP16 everywhere; auto-opt = INT8 backbone + FP16 rest (default).",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/ov",
        help="Directory to write the .xml/.bin (default: checkpoints/ov).",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Output filename without extension (default: dfine_<model>_coco<suffix>).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square input image size used for tracing (default: 640).",
    )
    parser.add_argument(
        "--topk-k",
        type=int,
        default=8,
        help="Classes kept per query by the two-stage TopK (default: 8).",
    )
    parser.add_argument(
        "--calib-images",
        type=int,
        default=300,
        help="Number of val images used to calibrate INT8 statistics (default: 300).",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        help="Device the INT8 scheme is tuned for, and the one --verify runs on " "(default: GPU).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare the written IR against PyTorch on one val image.",
    )
    args = parser.parse_args()

    if args.config is None or args.checkpoint_path is None:
        if args.model is None:
            parser.error("Provide either --model {n,s,m,l,x} or both --config and --checkpoint-path.")
        if args.config is None:
            args.config = f"configs/dfine/dfine_hgnetv2_{args.model}_coco.yml"
        if args.checkpoint_path is None:
            args.checkpoint_path = f"checkpoints/dfine_{args.model}_coco.pth"

    if args.basename is None:
        stem = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        args.basename = stem + SUFFIX[args.precision]

    return args


def main():
    args = parse_args()

    print(f"OpenVINO version : {ov.__version__}")
    print(f"Config           : {args.config}")
    print(f"Checkpoint       : {args.checkpoint_path}")
    print(f"Precision        : {args.precision}")

    model, cfg = build_deploy_model(args.config, args.checkpoint_path)

    print("Converting to OpenVINO IR ...")
    ov_model = convert(model, args.imgsz)

    if args.precision == "auto-opt":
        ov_model = build_auto_opt(ov_model, cfg, args)

    os.makedirs(args.output_dir, exist_ok=True)
    xml_path = os.path.join(args.output_dir, args.basename + ".xml")
    bin_path = os.path.join(args.output_dir, args.basename + ".bin")

    save_ir(ov_model, xml_path, args.topk_k)
    print(f"Saved IR         : {xml_path} ({os.path.getsize(bin_path) / 1e6:.1f} MB)")

    if args.verify:
        verify(xml_path, model, cfg, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
