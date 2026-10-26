"""Shared helpers for the ByteTrack (YOLOX-X + BYTETracker) benchmarking / accuracy / export scripts.

Everything that is *not* in the network weights (pre-processing, post-processing, tracker
parameters) lives here and is shared by the PyTorch and OpenVINO paths so that accuracy
comparisons only differ in the network execution backend/precision.
"""
import glob
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))                     # .../Tracking-Algorithms/BYTE-TRACK
REPO = os.path.join(ROOT, "ByteTrack_repo")
_W = "bytetrack_x_mot17.pth.tar"
WEIGHTS = os.environ.get("BYTETRACK_WEIGHTS") or next(
    (p for p in (os.path.join(ROOT, "weights", _W), os.path.join(ROOT, "..", "weights", _W)) if os.path.exists(p)),
    os.path.join(ROOT, "weights", _W))
WEIGHTS = os.path.abspath(WEIGHTS)
MOT17_TRAIN = os.path.join(ROOT, "datasets", "MOT17", "train")
OV_DIR = os.path.join(ROOT, "openvino_ir")
RESULTS = os.path.join(ROOT, "results")
TRACKER_CFG = os.path.join(OV_DIR, "bytetrack_tracker_config.json")

INPUT_HW = (800, 1440)
STRIDES = (8, 16, 32)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# numpy>=1.24 removed np.float (still used by ByteTrack's matching.py)
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
# numpy 2.x removed np.asfarray (still used by motmetrics 1.4)
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_tracker_cfg():
    with open(TRACKER_CFG) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------
def build_yolox_x(ckpt=WEIGHTS):
    import torch
    import torch.nn as nn
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

    depth, width, in_ch = 1.33, 1.25, [256, 512, 1024]
    model = YOLOX(YOLOPAFPN(depth, width, in_channels=in_ch), YOLOXHead(1, width, in_channels=in_ch))
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd)
    model.eval()
    return model


def build_detector(fuse=True, ckpt=WEIGHTS, input_hw=INPUT_HW):
    """YOLOX-X with the box decode folded into the graph (same graph used for PyTorch and OpenVINO).

    Output: [B, A, 6] = (cx, cy, w, h, obj, cls) in letterboxed-input pixel space, A=23625 @800x1440.
    """
    import torch
    import torch.nn as nn
    from yolox.utils import fuse_model

    class YoloxDecoded(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.model.head.decode_in_inference = False
            grids, strides = [], []
            for s in STRIDES:
                h, w = input_hw[0] // s, input_hw[1] // s
                yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
                grids.append(torch.stack((xv, yv), 2).view(1, -1, 2))
                strides.append(torch.full((1, h * w, 1), s))
            self.register_buffer("grids", torch.cat(grids, 1).float())
            self.register_buffer("strides", torch.cat(strides, 1).float())

        def forward(self, x):
            out = self.model(x)
            xy = (out[..., :2] + self.grids) * self.strides
            wh = torch.exp(out[..., 2:4]) * self.strides
            return torch.cat([xy, wh, out[..., 4:]], dim=-1)

    model = build_yolox_x(ckpt)
    if fuse:
        model = fuse_model(model)
    return YoloxDecoded(model).eval()


# --------------------------------------------------------------------------------------------
# Pre / post processing (identical to yolox.data.data_augment.preproc + yolox.utils.postprocess)
# --------------------------------------------------------------------------------------------
def preprocess(img_bgr, input_hw=INPUT_HW):
    import cv2

    h, w = img_bgr.shape[:2]
    r = min(input_hw[0] / h, input_hw[1] / w)
    resized = cv2.resize(img_bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded = np.full((input_hw[0], input_hw[1], 3), 114.0, dtype=np.float32)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    padded = padded[:, :, ::-1] / 255.0
    padded = (padded - MEAN) / STD
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32), r


def postprocess(pred, conf_thre=0.01, nms_thre=0.7):
    """pred: np.ndarray [1, A, 6] float -> torch.Tensor [K, 7] (x1,y1,x2,y2,obj,cls,cls_id) or None."""
    import torch
    from yolox.utils import postprocess as yolox_postprocess

    t = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32))
    return yolox_postprocess(t, 1, conf_thre, nms_thre)[0]


# --------------------------------------------------------------------------------------------
# MOT17 val-half (ByteTrack ablation protocol: second half of each MOT17 train FRCNN sequence)
# --------------------------------------------------------------------------------------------
def mot17_val_half(root=MOT17_TRAIN, seqs=None):
    out = []
    for seq_dir in sorted(glob.glob(os.path.join(root, "MOT17-*-FRCNN"))):
        name = os.path.basename(seq_dir)
        if seqs and name not in seqs:
            continue
        imgs = sorted(glob.glob(os.path.join(seq_dir, "img1", "*.jpg")))
        n = len(imgs)
        start = n // 2 + 1                                  # 0-based index of first val frame
        frames = [(i + 1 - start, os.path.join(seq_dir, "img1", f"{i + 1:06d}.jpg")) for i in range(start, n)]
        out.append({"name": name, "dir": seq_dir, "start": start, "frames": frames})
    return out


def write_gt_val_half(seq, dst):
    gt = np.loadtxt(os.path.join(seq["dir"], "gt", "gt.txt"), delimiter=",", dtype=np.float32)
    lo, hi = seq["start"], seq["start"] + len(seq["frames"]) - 1
    keep = (gt[:, 0] - 1 >= lo) & (gt[:, 0] - 1 <= hi)
    gt = gt[keep]
    gt[:, 0] -= lo
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        for o in gt:
            f.write("{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n".format(*[int(v) for v in o[:8]], o[8]))


def tracker_args(seq_name, cfg=None):
    cfg = cfg or load_tracker_cfg()
    t = cfg["tracker"]
    a = types.SimpleNamespace(track_thresh=t["track_thresh"], track_buffer=t["track_buffer"],
                              match_thresh=t["match_thresh_first_assoc"], min_box_area=t["min_box_area"],
                              mot20=t["mot20"])
    for k, v in cfg["per_sequence_overrides_mot17"].get(seq_name, {}).items():
        setattr(a, k, v)
    return a


def write_results(filename, results):
    with open(filename, "w") as f:
        for frame_id, tlwhs, ids, scores in results:
            for (x1, y1, w, h), tid, s in zip(tlwhs, ids, scores):
                if tid < 0:
                    continue
                f.write(f"{frame_id},{tid},{round(x1, 1)},{round(y1, 1)},{round(w, 1)},{round(h, 1)},{round(s, 2)},-1,-1,-1\n")


def system_info():
    import platform
    info = {"cpu": platform.processor()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo") as f:
            info["mem_total_gb"] = round(int(f.readline().split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return info
