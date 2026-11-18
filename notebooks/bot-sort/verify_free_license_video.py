"""Verify the free-license replacement video works with the BoT-SORT OpenVINO pipeline.

Mirrors bot-sort.ipynb: build PyTorch nets from local checkpoints, export FP16 IR,
track the first 120 frames of dog_walker_tokyo.mp4 (CC BY-SA 4.0, Wikimedia Commons).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import openvino as ov
import torch

import botsort_nb_utils as U

REPO = Path("/home/ubuntu/MultiObjectTrcking/BoT-SORT")
DET_W = REPO / "pretrained/bytetrack_x_mot17.pth.tar"
REID_W = REPO / "pretrained/mot17_sbs_S50.pth"
VIDEO = Path(__file__).parent / "dog_walker_tokyo.mp4"
OUT = Path(__file__).parent / "results/video"

U.setup_environment(REPO)
cfg = U.DEFAULT_CONFIG

print("building detector...")
det_pt = U.build_detector(DET_W)
print("building reid...")
reid_pt = U.build_reid(REID_W, repo=REPO)

DET_SHAPE = [1, 3, *cfg["detector"]["input_size_hw"]]
REID_SHAPE = [cfg["reid"]["batch"], 3, *cfg["reid"]["input_hw"]]

core = ov.Core()
IR = Path("ov_models_test")
IR.mkdir(exist_ok=True)
det_xml, reid_xml = IR / "det.xml", IR / "reid.xml"
if not det_xml.exists():
    m = ov.convert_model(det_pt, example_input=torch.zeros(DET_SHAPE), input=[ov.PartialShape(DET_SHAPE)])
    ov.save_model(m, str(det_xml), compress_to_fp16=True)
if not reid_xml.exists():
    m = ov.convert_model(reid_pt, example_input=torch.zeros(REID_SHAPE), input=[ov.PartialShape(REID_SHAPE)])
    ov.save_model(m, str(reid_xml), compress_to_fp16=True)

det_run = U.ov_runner(core, det_xml, "CPU")
reid_run = U.ov_runner(core, reid_xml, "CPU")

t0 = time.perf_counter()
r = U.track_video(det_run, reid_run, cfg, "free_license_check", video=VIDEO,
                  out_dir=OUT, max_frames=120)
s = r["stats"]
print(json.dumps(s, indent=2, default=str))
print(f"wall: {time.perf_counter()-t0:.1f}s")
print("mp4:", s.get("mp4"))
