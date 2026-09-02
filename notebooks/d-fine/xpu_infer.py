#!/usr/bin/env python
#
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
D-FINE XPU (Intel GPU) inference / evaluation script.

Runs COCO evaluation using torch device="xpu" (eager mode).
Reuses the existing evaluation pipeline (src.solver.det_engine.evaluate).

Examples
--------
    python xpu_infer.py --model n
    python xpu_infer.py --model x --batch-size 8
    python xpu_infer.py -c configs/dfine/dfine_hgnetv2_l_coco.yml -r checkpoints/dfine_l_coco.pth
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import YAMLConfig
from src.solver.det_engine import evaluate

MODELS = ("n", "s", "m", "l", "x")


def parse_args():
    parser = argparse.ArgumentParser(description="D-FINE XPU inference (COCO evaluation)")
    parser.add_argument(
        "--model",
        choices=MODELS,
        default=None,
        help="Model variant. Maps to configs/dfine/dfine_hgnetv2_<model>_coco.yml "
        "and checkpoints/dfine_<model>_coco.pth.",
    )
    parser.add_argument("-c", "--config", default=None, help="Explicit config yaml path.")
    parser.add_argument("-r", "--resume", default=None, help="Explicit checkpoint (.pth) path.")
    parser.add_argument("--device", default="xpu", help="Torch device (default: xpu).")
    parser.add_argument("--batch-size", type=int, default=8, help="Val batch size (default: 8).")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (default: 0).")
    args = parser.parse_args()

    if args.config is None or args.resume is None:
        if args.model is None:
            parser.error("Provide either --model {n,s,m,l,x} or both --config and --resume.")
        if args.config is None:
            args.config = f"configs/dfine/dfine_hgnetv2_{args.model}_coco.yml"
        if args.resume is None:
            args.resume = f"checkpoints/dfine_{args.model}_coco.pth"
    return args


def build_config(args):
    # Overrides into the yaml config.
    update = {
        "val_dataloader": {
            "num_workers": args.num_workers,
            "total_batch_size": args.batch_size,
        },
    }
    cfg = YAMLConfig(args.config, **update)

    # Skip fetching ImageNet pretrained backbone weights; the checkpoint has everything.
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    return cfg


def load_checkpoint(cfg, resume):
    checkpoint = torch.load(resume, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    # Strip potential "module." prefixes from DDP-saved checkpoints.
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    cfg.model.load_state_dict(state)

    return cfg.model


def main():
    args = parse_args()

    if args.device.startswith("xpu"):
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError(
                "torch.xpu is not available. Ensure the torch XPU build and Intel GPU "
                "runtime are installed and an Intel GPU is visible."
            )
        print(f"Using XPU device: {torch.xpu.get_device_name(0)}")

    device = torch.device(args.device)

    print(f"Config     : {args.config}")
    print(f"Checkpoint : {args.resume}")

    cfg = build_config(args)

    model = load_checkpoint(cfg, args.resume).to(device)
    model.eval()

    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)

    val_dataloader = cfg.val_dataloader
    evaluator = cfg.evaluator

    with torch.no_grad():
        evaluate(
            model,
            criterion,
            postprocessor,
            val_dataloader,
            evaluator,
            device,
            epoch=-1,
            use_wandb=False,
        )


if __name__ == "__main__":
    main()
