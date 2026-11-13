#!/usr/bin/env python
"""Write an input dump for `export_openvino.py --ref` for ANY task, without building the model.

`scripts/run_infer_smoke.py --dump` is the normal producer, but it runs a full inference first,
which needs a 11 GB checkpoint load (~100 s of md5 re-hashing) and, for a second task, torch
weights we do not need here: the export path only consumes `dump["inputs"]` (it re-derives every
torch reference itself in `derive_*_ref`).  Shapes come from the hydra task config through
`measure_latency.make_run_spec`, so `--task robotwin_...` yields input_image [1,3,384,320] and
proprio [1,14] instead of libero's [1,3,224,448] / [1,8].

Usage:
    FasterWAM/.venv/bin/python scripts/make_ref.py \
        --task robotwin_fasterwam_3cam_384_1e-4 --out scripts/artifacts/robotwin/ref_robotwin.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(SCRIPTS))

import run_infer_smoke as RIS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fasterwam")
    p.add_argument("--task", default=None)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0, help="synthetic seed for the random image")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.set_grad_enabled(False)
    cfg = RIS.load_cfg(args.model, args.task)
    _ns, spec = RIS.make_spec(args.model, cfg, args.task, action_horizon=None, num_inference_steps=args.steps, synthetic_seed=args.seed)
    print(f"[ref] spec {json.dumps(spec.__dict__, default=str)}", flush=True)
    inputs = RIS.prepare_inputs(spec, torch.device("cpu"))
    inputs = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inputs.items()}
    for k, v in inputs.items():
        print(f"[ref] {k}: {list(v.shape)} {v.dtype}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "inputs": inputs,
            "spec": spec.__dict__,
            "kwargs": {
                "prompt": None,
                "action_horizon": int(spec.action_horizon),
                "num_video_frames": int(spec.num_video_frames),
                "num_inference_steps": int(args.steps),
                "seed": 42,
                "rand_device": "cpu",
                "tiled": False,
            },
            "note": "shape-only input dump for export --ref (no action reference stored)",
        },
        str(out),
    )
    print(f"[ref] wrote {out} ({out.stat().st_size / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
