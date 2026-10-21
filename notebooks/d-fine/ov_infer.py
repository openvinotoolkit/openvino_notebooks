#!/usr/bin/env python
#
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
COCO evaluation for the IRs written by ov_export.py.

Compiles with the f16 inference hint and picks which exported IR to load.
Detections go through the same CocoEvaluator train.py --test-only uses.

Examples
--------
    cd /workspace

    python ov_infer.py --model n --precision auto-opt
    python ov_infer.py --model n --precision fp16 --device GPU --batch-size 8
    python ov_infer.py --ir checkpoints/ov/dfine_l_coco_auto_opt.xml \
        -c configs/dfine/dfine_hgnetv2_l_coco.yml
"""

import argparse
import os
import sys

import numpy as np
import openvino as ov
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import YAMLConfig
from src.data.dataset import mscoco_label2category
from ov_export import IMAGE_CHANNELS, SUFFIX, input_shapes

MODELS = ("n", "s", "m", "l", "x")


def parse_args():
    parser = argparse.ArgumentParser(description="D-FINE OpenVINO inference (COCO evaluation, f16 hint)")
    parser.add_argument(
        "--model",
        choices=MODELS,
        default=None,
        help="Model variant. Maps to configs/dfine/dfine_hgnetv2_<model>_coco.yml " "and checkpoints/ov/dfine_<model>_coco<suffix>.xml.",
    )
    parser.add_argument("-c", "--config", default=None, help="Explicit config yaml path.")
    parser.add_argument("--ir", default=None, help="Explicit OpenVINO IR .xml path.")
    parser.add_argument(
        "--precision",
        choices=tuple(SUFFIX),
        default="auto-opt",
        help="Which exported IR to evaluate (default: auto-opt).",
    )
    parser.add_argument(
        "--ir-dir",
        default="checkpoints/ov",
        help="Directory holding the exported IRs (default: checkpoints/ov).",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        help="OpenVINO device: GPU or CPU (default: GPU).",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Val batch size (default: 4).")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers (default: 4).")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square input image size (default: 640).",
    )
    args = parser.parse_args()

    if args.config is None:
        if args.model is None:
            parser.error("Provide either --model {n,s,m,l,x} or --config (with --ir).")
        args.config = f"configs/dfine/dfine_hgnetv2_{args.model}_coco.yml"
    if args.ir is None:
        if args.model is None:
            parser.error("Provide either --model {n,s,m,l,x} or --ir.")
        args.ir = os.path.join(args.ir_dir, f"dfine_{args.model}_coco{SUFFIX[args.precision]}.xml")

    return args


def load_ov_model(args, core):
    """Read the exported IR, pin it to the eval batch, and compile it with the f16 hint."""
    if not os.path.isfile(args.ir):
        raise FileNotFoundError(f"IR '{args.ir}' not found; export it first with " f"ov_export.py --precision {args.precision}.")

    print(f"IR             : {args.ir}")
    model = core.read_model(args.ir)

    model.reshape(input_shapes(args.batch_size, args.imgsz))
    print(f"Input shape    : [{args.batch_size}, {IMAGE_CHANNELS}, " f"{args.imgsz}, {args.imgsz}] (static)")

    return core.compile_model(model, args.device, {"INFERENCE_PRECISION_HINT": "f16"})


def build_config(args):
    update = {
        "val_dataloader": {
            "num_workers": args.num_workers,
            "total_batch_size": args.batch_size,
        },
    }
    cfg = YAMLConfig(args.config, **update)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    return cfg


def remap_labels(labels):
    """Map contiguous model label indices (0..79) to COCO category ids."""
    flat = labels.reshape(-1)
    mapped = np.fromiter((mscoco_label2category[int(x)] for x in flat), dtype=np.int64, count=flat.size)
    return mapped.reshape(labels.shape)


def main():
    args = parse_args()

    print(f"Config           : {args.config}")
    print(f"Device           : {args.device}  (precision: {args.precision}, hint: f16)")

    core = ov.Core()
    if args.device not in core.available_devices:
        raise RuntimeError(f"Device '{args.device}' not available. Present: {core.available_devices}")
    print(f"Device name      : {core.get_property(args.device, 'FULL_DEVICE_NAME')}")

    compiled = load_ov_model(args, core)
    out_labels = compiled.output("labels")
    out_boxes = compiled.output("boxes")
    out_scores = compiled.output("scores")

    cfg = build_config(args)
    val_dataloader = cfg.val_dataloader
    coco_evaluator = cfg.evaluator
    coco_evaluator.cleanup()

    n_images = 0
    for samples, targets in val_dataloader:
        images = np.ascontiguousarray(samples.numpy(), dtype=np.float32)
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        sizes = np.ascontiguousarray(orig_sizes.numpy(), dtype=np.int64)

        short = args.batch_size - images.shape[0]
        if short > 0:
            images = np.concatenate([images, np.repeat(images[-1:], short, 0)], 0)
            sizes = np.concatenate([sizes, np.repeat(sizes[-1:], short, 0)], 0)

        result = compiled({"images": images, "orig_sizes": sizes})

        labels = remap_labels(result[out_labels])
        boxes = result[out_boxes]
        scores = result[out_scores]

        res = {}
        for i, target in enumerate(targets):
            res[target["image_id"].item()] = {
                "labels": torch.as_tensor(labels[i]),
                "boxes": torch.as_tensor(boxes[i]),
                "scores": torch.as_tensor(scores[i]),
            }
        coco_evaluator.update(res)
        n_images += len(targets)
        if n_images % 500 == 0:
            print(f"  processed {n_images} images ...", flush=True)

    print(f"Evaluated {n_images} images.\n")
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    ap_val = float(coco_evaluator.coco_eval["bbox"].stats[0])

    print("\n=== Results ===")
    label = f"D-FINE-{args.model.upper()}" if args.model else os.path.basename(args.ir)
    print(f"Model            : {label}   (device={args.device}, batch={args.batch_size}, " f"precision={args.precision})")
    print(f"AP^val (0.50:0.95): {ap_val * 100:.1f}")


if __name__ == "__main__":
    main()
