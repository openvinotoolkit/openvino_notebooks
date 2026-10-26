"""Detector/ReID conversion and host-side tracking for bot-sort.ipynb.

BoT-SORT (https://github.com/NirAharon/BoT-SORT, https://arxiv.org/abs/2206.14651) = YOLOX-X detector + FastReID
SBS-S50 appearance network + BoT-SORT tracker (camera-motion compensation, Kalman filter, IoU + ReID fusion).
Everything that is not a network (pre/post-processing, crops, GMC, tracker,
MOT17 protocol and video test) lives here, so every OpenVINO precision runs
the same host pipeline; it mirrors the official ``tools/track.py``.

All paths are relative to this folder (``notebooks/``) and its parent; large inputs are looked up in the repository
and its parent folder and can be overridden with environment variables:
    BOTSORT_REPO, BOTSORT_DET_WEIGHTS, BOTSORT_REID_WEIGHTS, MOT17_TRAIN, VIDEO_1, VIDEO_2
"""

import glob
import json
import os
import re
import sys
import threading
import time
import types
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore")  # torch.jit / .grad warnings print site-packages paths

# ----------------------------------------------------------------------------------------------------------- paths
NB_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BOTSORT_PROJECT", NB_DIR)).expanduser().resolve()
ARTIFACTS = NB_DIR / "artifacts"  # hand-over PyTorch notebook -> OpenVINO notebook
REPO_URL = "https://github.com/NirAharon/BoT-SORT.git"
REPO = Path(os.environ.get(
    "BOTSORT_REPO",
    ROOT / "BoT-SORT_repo" if (ROOT / "BoT-SORT_repo").is_dir() else ROOT / "model" / "BoT-SORT_repo",
))

# official weights (Google Drive ids from the ByteTrack / BoT-SORT model zoos)
DET_WEIGHTS_GDRIVE = "1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5"  # bytetrack_x_mot17.pth.tar (BoT-SORT uses ByteTrack's MOT17 YOLOX-X)
REID_WEIGHTS_GDRIVE = "1QZFWpoa80rqo7O-HXmlss8J8CnS7IUsN"  # mot17_sbs_S50.pth (FastReID SBS-S50 trained on MOT17)


def find_file(name: str, env: Optional[str] = None, subdirs: Tuple[str, ...] = ("", "weights", "pretrained", "videos", "data")) -> Path:
    """First existing ``<base>/<subdir>/<name>`` for base in (notebooks/, repo root, repo parent); ``env`` overrides.

    :param name: file name to look for
    :param env: environment variable that, if set, gives the path directly
    :param subdirs: sub-folders searched under every base folder
    :returns: resolved path (or ``<repo root>/weights/<name>`` if nothing exists yet)
    """
    if env and os.environ.get(env):
        return Path(os.environ[env]).expanduser().resolve()
    for base in (NB_DIR, ROOT, ROOT.parent):
        for sd in subdirs:
            p = base / sd / name
            if p.exists():
                return p.resolve()
    return (ROOT if ROOT != NB_DIR else NB_DIR / "model") / "weights" / name


def find_mot17() -> Path:
    """MOT17 ``train`` folder: $MOT17_TRAIN, ``<repo>/datasets/MOT17/train``, or any ``<parent>/*/datasets/MOT17/train``."""
    if os.environ.get("MOT17_TRAIN"):
        return Path(os.environ["MOT17_TRAIN"]).expanduser().resolve()
    cands = [ROOT / "datasets/MOT17/train", NB_DIR / "data/MOT17/train"]
    return next((p.resolve() for p in cands if p.exists()), NB_DIR / "data/MOT17/train")


def rel(p) -> str:
    """Path relative to the notebooks folder (for printing / storing without machine-specific prefixes)."""
    try:
        return os.path.relpath(str(p), str(NB_DIR))
    except ValueError:
        return str(p)


DET_WEIGHTS = find_file("bytetrack_x_mot17.pth.tar", "BOTSORT_DET_WEIGHTS")
REID_WEIGHTS = find_file("mot17_sbs_S50.pth", "BOTSORT_REID_WEIGHTS")
MOT17_TRAIN = find_mot17()
VIDEO_1 = find_file("pedestrians_cc0_30s.mp4", "VIDEO_1")
VIDEO_2 = find_file("Luka_Doncic.mp4", "VIDEO_2")

# ----------------------------------------------------------------------------------------------------------- config
# NOT in either weight file: needed identically by the PyTorch and the OpenVINO pipeline (tools/track.py,
# --default-parameters --with-reid --fuse, MOT17 val = ablation protocol; tools/demo.py defaults for videos).
DEFAULT_CONFIG = {
    "detector": {"input_size_hw": [800, 1440], "strides": [8, 16, 32], "depth": 1.33, "width": 1.25, "num_classes": 1,
                 "conf_thresh": 0.09, "nms_iou_thresh": 0.7, "_conf": "max(0.001, track_low_thresh - 0.01)"},
    "preprocess": {"resize": "letterbox keep-ratio, int() sizes, top-left", "pad_value": 114.0, "color": "BGR->RGB", "scale": 1 / 255.0,
                   "mean_rgb": [0.485, 0.456, 0.406], "std_rgb": [0.229, 0.224, 0.225]},
    "reid": {"config": "fast_reid/configs/MOT17/sbs_S50.yml", "input_hw": [384, 128], "batch": 8, "embedding_dim": 2048,
             "crop": "int box, clip to image, BGR->RGB, cv2.resize to 128x384 (no letterbox), float 0..255 (mean/std inside the network)",
             "post": "inf -> 1.0, L2 normalisation"},
    "tracker": {"track_high_thresh": 0.6, "track_low_thresh": 0.1, "new_track_thresh": 0.7, "track_buffer": 30, "match_thresh": 0.8,
                "proximity_thresh": 0.5, "appearance_thresh": 0.25, "mot20": False, "cmc_method": "file", "frame_rate": 30,
                "min_box_area": 10, "max_aspect_ratio_w_over_h": 1.6},
    "per_sequence_overrides_mot17": {
        **{f"MOT17-{s}-FRCNN": {"gmc_file": f"MOT17_ablation/GMC-MOT17-{s}.txt"} for s in ("02", "04", "09", "10", "11")},
        "MOT17-05-FRCNN": {"gmc_file": "MOT17_ablation/GMC-MOT17-05.txt", "track_buffer": 14},
        "MOT17-13-FRCNN": {"gmc_file": "MOT17_ablation/GMC-MOT17-13.txt", "track_buffer": 25},
    },
    "video_tracker": {"cmc_method": "orb", "_frame_rate": "video fps"},
}


# ----------------------------------------------------------------------------------------------------------- environment
def _bbox_overlaps(boxes, query_boxes):
    """numpy port of ``cython_bbox.bbox_overlaps`` (not buildable with numpy 2), same +1 pixel convention."""
    boxes, query_boxes = np.asarray(boxes, np.float64), np.asarray(query_boxes, np.float64)
    if len(boxes) == 0 or len(query_boxes) == 0:
        return np.zeros((len(boxes), len(query_boxes)))
    qa = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    ba = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    iw = np.minimum(boxes[:, None, 2], query_boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], query_boxes[None, :, 0]) + 1
    ih = np.minimum(boxes[:, None, 3], query_boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], query_boxes[None, :, 1]) + 1
    ok = (iw > 0) & (ih > 0)
    inter = np.where(ok, iw * ih, 0.0)
    return np.where(ok, inter / (ba[:, None] + qa[None, :] - inter), 0.0)


def _lapjv(cost, extend_cost=True, cost_limit=np.inf, return_cost=True):
    """scipy stand-in for ``lap.lapjv`` with ``cost_limit`` (only used if the ``lap`` package is missing)."""
    from scipy.optimize import linear_sum_assignment

    cost = np.asarray(cost, np.float64)
    n, m = cost.shape
    x, y = -np.ones(n, int), -np.ones(m, int)
    if n and m:
        if np.isfinite(cost_limit):
            ext = np.full((n + m, m + n), 1e9)
            ext[:n, :m] = cost
            ext[:n, m:][np.arange(n), np.arange(n)] = cost_limit / 2.0
            ext[n:, :m][np.arange(m), np.arange(m)] = cost_limit / 2.0
            ext[n:, m:] = 0.0
            r, c = linear_sum_assignment(ext)
            for i, j in zip(r, c):
                if i < n and j < m:
                    x[i], y[j] = j, i
        else:
            r, c = linear_sum_assignment(cost)
            x[r], y[c] = c, r
    return float(sum(cost[i, x[i]] for i in range(n) if x[i] >= 0)), x, y


def setup_environment(repo: Path = REPO) -> None:
    """Make the unmodified BoT-SORT repo importable on numpy 2 / any PyTorch device.

    * numpy 2 shims (``np.float``, ``np.float_``, ``np.asfarray``) used by the 2022 tracker code and motmetrics
    * ``cython_bbox`` / ``lap`` fall-backs if they are not installed
    * ``tracker/bot_sort.py`` imports ``FastReIDInterface`` at module level, which imports FastReID's training engine
      (faiss, tensorboard, ``torch._six``). We bring our own ReID encoder, so that module is stubbed.
    """
    import importlib.util

    for name, val in (("float", float), ("float_", np.float64)):
        if not hasattr(np, name):
            setattr(np, name, val)
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
    if importlib.util.find_spec("cython_bbox") is None and "cython_bbox" not in sys.modules:
        m = types.ModuleType("cython_bbox")
        m.bbox_overlaps = _bbox_overlaps
        sys.modules["cython_bbox"] = m
    if importlib.util.find_spec("lap") is None and "lap" not in sys.modules:
        m = types.ModuleType("lap")
        m.lapjv = _lapjv
        sys.modules["lap"] = m
    if "fast_reid.fast_reid_interfece" not in sys.modules:
        stub = types.ModuleType("fast_reid.fast_reid_interfece")
        stub.FastReIDInterface = None
        sys.modules["fast_reid.fast_reid_interfece"] = stub
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


# ----------------------------------------------------------------------------------------------------------- networks
def build_detector(weights: Optional[Path] = None, input_hw=(800, 1440), cfg: dict = DEFAULT_CONFIG):
    """YOLOX-X exactly as ``yolox/exps/example/mot/yolox_x_mix_det.py``, Conv+BN fused (``--fuse``), with the
    anchor-free grid decode folded into the graph -> output ``[1, A, 6] = (cx, cy, w, h, obj, cls)`` in input pixels.

    :param weights: ``bytetrack_x_mot17.pth.tar``; ``None`` builds the fused structure only (to load a state_dict)
    """
    import torch
    import torch.nn as nn
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
    from yolox.utils import fuse_model

    d = cfg["detector"]
    in_ch = [256, 512, 1024]
    model = YOLOX(YOLOPAFPN(d["depth"], d["width"], in_channels=in_ch), YOLOXHead(d["num_classes"], d["width"], in_channels=in_ch))
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["model"])
    model = fuse_model(model.eval())
    return YoloxDecoded(model, input_hw, tuple(d["strides"])).eval()


def _decoded_cls():
    import torch
    import torch.nn as nn

    class YoloxDecoded(nn.Module):
        """YOLOX + grid / stride decode inside the graph (same graph for PyTorch and OpenVINO)."""

        def __init__(self, model, input_hw=(800, 1440), strides=(8, 16, 32)):
            super().__init__()
            self.model = model
            self.model.head.decode_in_inference = False
            grids, strs = [], []
            for s in strides:
                h, w = input_hw[0] // s, input_hw[1] // s
                yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
                grids.append(torch.stack((xv, yv), 2).view(1, -1, 2))
                strs.append(torch.full((1, h * w, 1), s))
            self.register_buffer("grids", torch.cat(grids, 1).float())
            self.register_buffer("strides", torch.cat(strs, 1).float())

        def forward(self, x):
            o = self.model(x)
            return torch.cat([(o[..., :2] + self.grids) * self.strides, torch.exp(o[..., 2:4]) * self.strides, o[..., 4:]], -1)

    return YoloxDecoded


def build_reid(weights: Optional[Path] = None, repo: Path = REPO, cfg: dict = DEFAULT_CONFIG):
    """FastReID SBS-S50 (ResNeSt-50 + non-local + GeM + BN neck) as ``FastReIDInterface`` builds it.

    FastReID's ``Baseline.preprocess_image`` normalises the input *in place* (``images.sub_(mean).div_(std)``); on CPU
    ``torch.from_numpy`` shares memory with the caller's array, which would then be corrupted -> out-of-place override
    (identical maths). ``weights=None`` builds the structure only (to load a state_dict).
    """
    import logging

    from fast_reid.fastreid.config import get_cfg
    from fast_reid.fastreid.modeling.meta_arch import build_model
    from fast_reid.fastreid.utils.checkpoint import Checkpointer

    c = get_cfg()
    c.merge_from_file(str(repo / cfg["reid"]["config"]))
    c.MODEL.BACKBONE.PRETRAIN = False
    c.MODEL.DEVICE = "cpu"
    c.freeze()
    model = build_model(c).eval()
    if weights is not None:
        logging.getLogger("fast_reid.fastreid.utils.checkpoint").setLevel(logging.ERROR)
        Checkpointer(model).load(str(weights))  # 'heads.weight' (training classifier) is skipped, as officially
    model.preprocess_image = types.MethodType(_preprocess_out_of_place, model)  # bound method: survives copy.deepcopy
    return model


def _preprocess_out_of_place(self, images):
    """Same maths as FastReID ``Baseline.preprocess_image`` without modifying ``images`` in place."""
    return (images - self.pixel_mean) / self.pixel_std


def torch_runner(model, device, dtype) -> Callable[[np.ndarray], np.ndarray]:
    """``run(x) -> np.ndarray``: host f32 input -> device (cast) -> forward -> f32 host output (like benchmark_app I/O)."""
    import torch

    m = model.to(device).to(dtype).eval()

    def run(x):
        with torch.inference_mode():
            return m(torch.from_numpy(x).to(device).to(dtype)).float().cpu().numpy()

    run.desc = f"torch-{device}-{str(dtype).split('.')[-1]}"
    return run


def ov_runner(core, xml, device: str = "GPU", **props) -> Callable[[np.ndarray], np.ndarray]:
    """Sync single-request OpenVINO runner (``PERFORMANCE_HINT=LATENCY``); ``run.exec_precision`` = GPU execution precision."""
    cm = core.compile_model(str(xml), device, {"PERFORMANCE_HINT": "LATENCY", **props})
    req = cm.create_infer_request()

    def run(x):
        req.infer({0: x})
        return req.get_output_tensor(0).data.copy()

    run.exec_precision = cm.get_property("INFERENCE_PRECISION_HINT").get_type_name() if device.startswith("GPU") else "n/a"
    run.desc = f"ov-{device}-{Path(xml).parent.name}-exec{run.exec_precision}"
    return run


def timeit(run: Callable, x: np.ndarray, n: int = 20, warmup: int = 3):
    """Average wall time of ``run(x)`` in ms (incl. host<->device copies)."""
    for _ in range(warmup):
        y = run(x)
    t = time.perf_counter()
    for _ in range(n):
        y = run(x)
    return y, (time.perf_counter() - t) * 1000 / n


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def score_cosine(a, b) -> float:
    """cosine of the detection score maps (obj x cls): what actually drives NMS / tracking."""
    return cosine(a[..., 4] * a[..., 5], b[..., 4] * b[..., 5])


# ----------------------------------------------------------------------------------------------------------- pre / post
def preprocess(img_bgr: np.ndarray, cfg: dict = DEFAULT_CONFIG) -> Tuple[np.ndarray, float]:
    """YOLOX ``preproc``: letterbox (int() sizes, top-left, pad 114), BGR->RGB, /255, ImageNet mean/std, NCHW f32."""
    H, W = cfg["detector"]["input_size_hw"]
    h, w = img_bgr.shape[:2]
    r = min(H / h, W / w)
    res = cv2.resize(img_bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    pad = np.full((H, W, 3), cfg["preprocess"]["pad_value"], np.float32)
    pad[: res.shape[0], : res.shape[1]] = res
    pad = (pad[:, :, ::-1] / 255.0 - np.array(cfg["preprocess"]["mean_rgb"], np.float32)) / np.array(cfg["preprocess"]["std_rgb"], np.float32)
    return np.ascontiguousarray(pad.transpose(2, 0, 1)[None], dtype=np.float32), r


def postprocess(pred: np.ndarray, ratio: float, cfg: dict = DEFAULT_CONFIG) -> np.ndarray:
    """score = obj x cls >= conf, class-aware NMS, boxes / letterbox ratio -> ``[N, 5]`` x1, y1, x2, y2, score (original pixels).

    ``yolox.utils.postprocess`` modifies its input in place, so the network output is copied first.
    """
    import torch
    from yolox.utils import postprocess as yolox_post

    p = torch.from_numpy(np.array(pred, dtype=np.float32, copy=True))
    d = yolox_post(p, p.shape[-1] - 5, cfg["detector"]["conf_thresh"], cfg["detector"]["nms_iou_thresh"])[0]
    if d is None:
        return np.zeros((0, 5), np.float32)
    d = d.numpy()
    return np.concatenate([d[:, :4] / ratio, (d[:, 4] * d[:, 5])[:, None]], 1).astype(np.float32)


def reid_crops(image_bgr: np.ndarray, tlbrs: np.ndarray, size_hw=(384, 128)) -> np.ndarray:
    """``FastReIDInterface.inference`` pre-processing: int box, clip, crop, BGR->RGB, resize to (W, H), float CHW 0..255."""
    H, W = image_bgr.shape[:2]
    out = []
    for d in np.asarray(tlbrs):
        b = d[:4].astype(np.int_)
        b[0], b[1], b[2], b[3] = max(0, b[0]), max(0, b[1]), min(W - 1, b[2]), min(H - 1, b[3])
        patch = image_bgr[b[1] : b[3], b[0] : b[2], :]
        if patch.size == 0:  # degenerate box (the official code would crash in cv2.resize)
            y, x = min(max(b[1], 0), H - 1), min(max(b[0], 0), W - 1)
            patch = image_bgr[y : y + 1, x : x + 1, :]
        patch = cv2.resize(patch[:, :, ::-1], (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
        out.append(patch.astype("float32").transpose(2, 0, 1))
    return np.stack(out) if out else np.zeros((0, 3, *size_hw), np.float32)


def batched(crops: np.ndarray, batch: int = 8):
    """Fixed-size batches (official batch size 8; last one zero-padded) -> yields ``(batch_array, n_valid)``."""
    for i in range(0, len(crops), batch):
        b = crops[i : i + batch]
        n = len(b)
        if n < batch:
            b = np.concatenate([b, np.zeros((batch - n, *b.shape[1:]), np.float32)])
        yield np.ascontiguousarray(b, dtype=np.float32), n


def normalize_feats(pred: np.ndarray) -> np.ndarray:
    pred = np.array(pred, dtype=np.float32, copy=True)
    pred[np.isinf(pred)] = 1.0
    return pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)


class ReIDEncoder:
    """Drop-in for ``FastReIDInterface``: ``inference(img, dets_tlbr) -> [N, 2048]`` L2-normalised embeddings.
    ``run(batch[8, 3, 384, 128]) -> [8, 2048]`` is a PyTorch or OpenVINO runner. Counts batches / crops / time."""

    def __init__(self, run: Callable, size_hw=(384, 128), batch: int = 8):
        self.run, self.size_hw, self.batch = run, size_hw, batch
        self.calls = self.crops = 0
        self.time_s = 0.0

    def inference(self, image, detections):
        if detections is None or np.size(detections) == 0:
            return []
        crops = reid_crops(image, detections, self.size_hw)
        feats = []
        t = time.perf_counter()
        for b, n in batched(crops, self.batch):
            feats.append(np.asarray(self.run(b))[:n])
            self.calls += 1
        self.time_s += time.perf_counter() - t
        self.crops += len(crops)
        return normalize_feats(np.concatenate(feats)).astype(np.float64)


# ----------------------------------------------------------------------------------------------------------- tracker
class FileGMC:
    """``tracker/gmc.py`` 'file' method (official precomputed camera motion) with a path independent of the CWD."""

    def __init__(self, path: Path):
        self.rows = [line.split("\t") for line in open(path) if line.strip()]
        self.i = 0

    def apply(self, raw_frame, detections=None):
        H = np.eye(2, 3, dtype=np.float64)
        if self.i < len(self.rows):
            H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2] = (float(v) for v in self.rows[self.i][1:7])
        self.i += 1
        return H


class SafeGMC:
    """Wraps the official ORB/SIFT GMC, which crashes (``for m, n in knnMatches``) when fewer than 2 neighbours are
    found on featureless frames (fades, black frames): identity warp for that frame, re-initialise on the next."""

    def __init__(self, gmc):
        self.gmc, self.failures, self.warnings = gmc, 0, 0

    def apply(self, raw_frame, detections=None):
        import contextlib
        import io

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):  # the official code prints 'Warning: not enough matching points' per frame
                return self.gmc.apply(raw_frame, detections)
        except (ValueError, IndexError, cv2.error):
            self.failures += 1
            self.gmc.initializedFirstFrame = False
            return np.eye(2, 3)
        finally:
            self.warnings += buf.getvalue().count("not enough matching points")


def tracker_params(cfg: dict, seq_name: Optional[str] = None, video: bool = False) -> dict:
    """Merged tracker params: MOT17 per-sequence overrides, or the generic-video profile."""
    p = dict(cfg["tracker"])
    if video:
        p.update({k: v for k, v in cfg.get("video_tracker", {}).items() if not k.startswith("_")})
    elif seq_name:
        p.update(cfg.get("per_sequence_overrides_mot17", {}).get(seq_name, {}))
    return p


def new_tracker(params: dict, frame_rate: int, encoder: Optional[ReIDEncoder], repo: Path = REPO):
    """Official ``BoTSORT`` with our ReID encoder and GMC (file / ORB / none)."""
    from tracker.basetrack import BaseTrack
    from tracker.bot_sort import BoTSORT

    BaseTrack.clear_count()  # global track-id counter -> reset for comparable runs
    ns = types.SimpleNamespace(
        track_high_thresh=params["track_high_thresh"], track_low_thresh=params["track_low_thresh"], new_track_thresh=params["new_track_thresh"],
        track_buffer=params["track_buffer"], match_thresh=params["match_thresh"], proximity_thresh=params["proximity_thresh"],
        appearance_thresh=params["appearance_thresh"], with_reid=False, cmc_method="none", name="", ablation=False, mot20=params["mot20"],
    )
    tr = BoTSORT(ns, frame_rate=frame_rate)
    if params["cmc_method"] == "file":
        tr.gmc = FileGMC(repo / "tracker" / "GMC_files" / params["gmc_file"])
    elif params["cmc_method"] != "none":
        from tracker.gmc import GMC

        tr.gmc = SafeGMC(GMC(method=params["cmc_method"], verbose=None))
    if encoder is not None:
        ns.with_reid = True
        tr.encoder = encoder
    return tr


def keep(tlwh, params: dict) -> bool:
    """Output filter of tools/track.py: min box area and w/h <= 1.6 (pedestrians)."""
    return tlwh[2] * tlwh[3] > params["min_box_area"] and tlwh[2] / tlwh[3] <= params["max_aspect_ratio_w_over_h"]


# ----------------------------------------------------------------------------------------------------------- MOT17
def mot17_val_half(seqs: Optional[List[str]] = None) -> List[dict]:
    """ByteTrack / BoT-SORT ablation protocol: second half of each MOT17 train FRCNN sequence."""
    out = []
    for d in sorted(MOT17_TRAIN.glob("MOT17-*-FRCNN")):
        if seqs and d.name not in seqs:
            continue
        n = len(list((d / "img1").glob("*.jpg")))
        start = n // 2 + 1
        out.append({"name": d.name, "dir": d, "start": start, "frames": [(i + 1 - start, d / "img1" / f"{i + 1:06d}.jpg") for i in range(start, n)]})
    return out


def train_half_frames() -> List[Path]:
    out = []
    for d in sorted(MOT17_TRAIN.glob("MOT17-*-FRCNN")):
        imgs = sorted((d / "img1").glob("*.jpg"))
        out += imgs[: len(imgs) // 2 + 1]
    return out


def write_gt_val_half(seq: dict, dst: Path) -> None:
    gt = np.loadtxt(seq["dir"] / "gt" / "gt.txt", delimiter=",", dtype=np.float32)
    lo, hi = seq["start"], seq["start"] + len(seq["frames"]) - 1
    gt = gt[(gt[:, 0] - 1 >= lo) & (gt[:, 0] - 1 <= hi)]
    gt[:, 0] -= lo
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for o in gt:
            f.write("{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n".format(*[int(v) for v in o[:8]], o[8]))


def track_sequence(det_run: Callable, encoder: Optional[ReIDEncoder], seq: dict, cfg: dict, max_frames: Optional[int] = None):
    """Track one sequence exactly like tools/track.py. Returns ``(rows, frames, detector_s, total_s)``;
    rows = ``(frame, id, tlwh, score)``."""
    params = tracker_params(cfg, seq["name"])
    tr = new_tracker(params, cfg["tracker"]["frame_rate"], encoder)
    frames = seq["frames"][:max_frames] if max_frames else seq["frames"]
    rows, t_det, t0 = [], 0.0, time.perf_counter()
    for fid, path in frames:
        img = cv2.imread(str(path))
        x, r = preprocess(img, cfg)
        a = time.perf_counter()
        pred = det_run(x)
        t_det += time.perf_counter() - a
        dets = postprocess(pred, r, cfg)
        for t in tr.update(dets, img):
            if keep(t.tlwh, params):
                rows.append((fid, t.track_id, t.tlwh.copy(), float(t.score)))
    return rows, len(frames), t_det, time.perf_counter() - t0


def mot_eval(det_run: Callable, reid_run: Optional[Callable], cfg: dict, tag: str, seqs: Optional[List[str]] = None,
             max_frames: Optional[int] = None, out_root: Path = Path("results/accuracy"), verbose: bool = False) -> dict:
    """Track MOT17 val-half with the given detector / ReID runners and score with motmetrics (as tools/track.py).

    :returns: metrics dict (MOTA, IDF1, IDs, FP, FN, ..., detector / ReID ms per frame, ReID batches per frame),
              also written to ``<out_root>/<tag>/metrics.json``
    """
    import motmetrics as mm

    mm.lap.default_solver = "lap"
    data = mot17_val_half(seqs)
    assert data, f"no MOT17 sequences in {rel(MOT17_TRAIN)} - see the 'Prerequisites' section"
    res_dir, gt_dir = Path(out_root) / tag / "track_results", Path(out_root) / "_gt_val_half"
    res_dir.mkdir(parents=True, exist_ok=True)
    accs, names, frames, t_det, t_all = [], [], 0, 0.0, 0.0
    enc = ReIDEncoder(reid_run, tuple(cfg["reid"]["input_hw"]), cfg["reid"]["batch"]) if reid_run else None
    for seq in data:
        write_gt_val_half(seq, gt_dir / f"{seq['name']}.txt")
        rows, n, td, ta = track_sequence(det_run, enc, seq, cfg, max_frames)
        with open(res_dir / f"{seq['name']}.txt", "w") as f:
            for fid, tid, (x, y, w, h), s in rows:
                f.write(f"{fid},{tid},{round(x, 1)},{round(y, 1)},{round(w, 1)},{round(h, 1)},{round(s, 2)},-1,-1,-1\n")
        gt = mm.io.loadtxt(str(gt_dir / f"{seq['name']}.txt"), fmt="mot15-2D", min_confidence=1)
        if max_frames:
            gt = gt[gt.index.get_level_values(0) <= max_frames]
        ts = mm.io.loadtxt(str(res_dir / f"{seq['name']}.txt"), fmt="mot15-2D", min_confidence=-1)
        accs.append(mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5))
        names.append(seq["name"])
        frames, t_det, t_all = frames + n, t_det + td, t_all + ta
        if verbose:
            print(f"  [{tag}] {seq['name']}: {n} frames", flush=True)
    s = mm.metrics.create().compute_many(accs, names=names, generate_overall=True,
                                         metrics=["mota", "idf1", "motp", "num_switches", "num_false_positives", "num_misses", "mostly_tracked", "mostly_lost"])
    o = s.loc["OVERALL"]
    m = {"tag": tag, "frames": frames, "sequences": names, "MOTA": 100 * float(o["mota"]), "IDF1": 100 * float(o["idf1"]),
         "MOTP": 100 * (1 - float(o["motp"])), "IDs": int(o["num_switches"]), "FP": int(o["num_false_positives"]), "FN": int(o["num_misses"]),
         "MT": int(o["mostly_tracked"]), "ML": int(o["mostly_lost"]), "detector_ms_per_frame": 1000 * t_det / frames,
         "reid_ms_per_frame": 1000 * enc.time_s / frames if enc else 0.0, "reid_batches_per_frame": enc.calls / frames if enc else 0.0,
         "reid_crops_per_frame": enc.crops / frames if enc else 0.0, "pipeline_ms_per_frame": 1000 * t_all / frames,
         "detector": getattr(det_run, "desc", ""), "reid": getattr(reid_run, "desc", "none"),
         "per_sequence": {k: {"MOTA": 100 * float(s.loc[k, "mota"]), "IDF1": 100 * float(s.loc[k, "idf1"])} for k in names}}
    json.dump(m, open(Path(out_root) / tag / "metrics.json", "w"), indent=2)
    return m


def accuracy_control_validation(reid_run: Callable, cfg: dict, seqs=("MOT17-09-FRCNN",), frames: int = 150):
    """``(validation_items, validation_fn)`` for ``nncf.quantize_with_accuracy_control`` on the detector: MOTA of the
    full BoT-SORT pipeline (candidate detector compiled by NNCF on CPU + fixed ReID runner) on a short val subset."""
    import motmetrics as mm
    import pandas as pd

    data = mot17_val_half(list(seqs))
    for s in data:
        s["frames"] = s["frames"][:frames]
    gt_dir = Path("results/accuracy/_gt_accuracy_control")

    def validation_fn(compiled_model, validation_subset) -> float:
        out = compiled_model.output(0)
        run = lambda x: np.asarray(compiled_model(x)[out])  # noqa: E731
        accs = []
        for seq in validation_subset:
            write_gt_val_half(seq, gt_dir / f"{seq['name']}.txt")
            enc = ReIDEncoder(reid_run, tuple(cfg["reid"]["input_hw"]), cfg["reid"]["batch"])
            rows, *_ = track_sequence(run, enc, seq, cfg)
            gt = mm.io.loadtxt(str(gt_dir / f"{seq['name']}.txt"), fmt="mot15-2D", min_confidence=1)
            gt = gt[gt.index.get_level_values(0) <= len(seq["frames"])]
            ts = pd.DataFrame([[f, i, *b, 1, -1, -1, -1] for f, i, b, _ in rows],
                              columns=["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence", "ClassId", "Visibility", "unused"]).set_index(["FrameId", "Id"])
            accs.append(mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5))
        mota = 100 * float(mm.metrics.create().compute_many(accs, metrics=["mota"], generate_overall=True).loc["OVERALL", "mota"])
        print(f"[NNCF] MOTA = {mota:.2f}", flush=True)
        return mota

    return data, validation_fn


# ----------------------------------------------------------------------------------------------------------- ReID data
def reid_gt_crops(frame_paths: Iterable[Path], batch: int = 8, size_hw=(384, 128), seed: int = 0):
    """Batches of ``batch`` ground-truth person crops (pedestrians, visibility >= 0.25) from MOT17 frames.
    Used for ReID NNCF calibration (train-half frames)."""
    rng, cache = np.random.default_rng(seed), {}
    for p in frame_paths:
        p = Path(p)
        seq_dir, fid = p.parent.parent, int(p.stem)
        if seq_dir not in cache:
            g = np.loadtxt(seq_dir / "gt" / "gt.txt", delimiter=",")
            cache[seq_dir] = g[(g[:, 6] == 1) & (g[:, 7] == 1) & (g[:, 8] >= 0.25)]
        g = cache[seq_dir][cache[seq_dir][:, 0] == fid]
        if len(g) == 0:
            continue
        g = g[rng.permutation(len(g))[:batch]]
        crops = reid_crops(cv2.imread(str(p)), np.stack([g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]], 1), size_hw)
        while len(crops) < batch:  # few people: repeat crops (no zero padding inside calibration data)
            crops = np.concatenate([crops, crops])[:batch]
        yield np.ascontiguousarray(crops[:batch], np.float32)


def _gt_crops_ids(path: Path, size_hw=(384, 128)):
    seq_dir, fid = Path(path).parent.parent, int(Path(path).stem)
    g = np.loadtxt(seq_dir / "gt" / "gt.txt", delimiter=",")
    g = g[(g[:, 0] == fid) & (g[:, 6] == 1) & (g[:, 7] == 1) & (g[:, 8] >= 0.25)]
    tlbr = np.stack([g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]], 1) if len(g) else np.zeros((0, 4))
    return reid_crops(cv2.imread(str(path)), tlbr, size_hw), g[:, 1].astype(int)


def _embed(run: Callable, crops: np.ndarray, batch: int = 8) -> np.ndarray:
    out = [np.asarray(run(b))[:n] for b, n in batched(crops, batch)]
    return normalize_feats(np.concatenate(out)) if out else np.zeros((0, 2048), np.float32)


def reid_fidelity(runners: Dict[str, Callable], pairs_per_seq: int = 6, gap: int = 15, batch: int = 8) -> Dict[str, dict]:
    """ReID quality per precision on MOT17 val-half GT crops: query = crops at frame f, gallery = crops at f + gap.

    rank-1 / mAP over identities, and embedding cosine vs the FIRST runner (the FP32 reference).
    """
    data = []
    for s in mot17_val_half():
        n = len(s["frames"])
        for f in np.linspace(0, n - 1 - gap, pairs_per_seq).round().astype(int):
            data.append((_gt_crops_ids(s["frames"][f][1]), _gt_crops_ids(s["frames"][f + gap][1])))
    res, ref = {}, None
    for name, run in runners.items():
        feats = [(_embed(run, q[0], batch), _embed(run, g[0], batch)) for q, g in data]
        r1, aps = [], []
        for (fq, fg), (q, g) in zip(feats, data):
            if len(fq) == 0 or len(fg) == 0:
                continue
            sim = fq @ fg.T
            for i in range(len(fq)):
                if q[1][i] not in g[1]:
                    continue
                match = g[1][np.argsort(-sim[i])] == q[1][i]
                r1.append(bool(match[0]))
                hits = np.where(match)[0]
                aps.append(np.mean([(k + 1) / (h + 1) for k, h in enumerate(hits)]))
        cos = None if ref is None else np.concatenate([np.sum(a * b, 1) for fa, fb in zip(feats, ref) for a, b in zip(fa, fb) if len(a)])
        res[name] = {"rank1": 100 * float(np.mean(r1)), "mAP": 100 * float(np.mean(aps)), "queries": len(r1),
                     "cos_mean": None if cos is None else float(cos.mean()), "cos_min": None if cos is None else float(cos.min()),
                     "crops": int(sum(len(q[0]) + len(g[0]) for q, g in data))}
        ref = ref or feats
    return res


# ----------------------------------------------------------------------------------------------------------- video
def _color(tid: int):
    return tuple(int(v) for v in np.random.default_rng(int(tid)).integers(60, 255, 3))


def draw_tracks(img: np.ndarray, tracks: Dict[int, np.ndarray], hud: Optional[str] = None) -> np.ndarray:
    out = img.copy()
    for tid, (x, y, w, h) in tracks.items():
        c = _color(tid)
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), c, 3)
        cv2.putText(out, str(tid), (int(x), max(20, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
    if hud:
        cv2.putText(out, hud, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    return out


def draw_dets(img: np.ndarray, dets: np.ndarray, thr: float = 0.6) -> np.ndarray:
    out = img.copy()
    for x1, y1, x2, y2, s in dets:
        if s >= thr:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
            cv2.putText(out, f"{s:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return out


def track_video(det_run: Callable, reid_run: Optional[Callable], cfg: dict, tag: str, video: Path = VIDEO_1,
                out_dir: Path = Path("results/video"), max_frames: Optional[int] = None, write_mp4: bool = True,
                cut_thresh: float = 35.0) -> dict:
    """Track a real video (no ground truth) with BoT-SORT's generic-video settings (ORB camera-motion compensation,
    tracker frame rate = video fps). Writes a half-resolution annotated MP4 and the tracks (JSON).

    :returns: ``{"tracks": [{id: tlwh}], "cuts": [frame ids], "stats": {...}}``; shot cut = mean |grey diff| of
              160x90 thumbnails > ``cut_thresh``
    """
    params = tracker_params(cfg, video=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H, N = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    N = min(N, max_frames) if max_frames else N
    enc = ReIDEncoder(reid_run, tuple(cfg["reid"]["input_hw"]), cfg["reid"]["batch"]) if reid_run else None
    tr = new_tracker(params, round(fps), enc)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{tag}_tracked.mp4"
    VW, VH = (W // 2) // 8 * 8, (H // 2) // 8 * 8
    writer = None
    if write_mp4:
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(str(mp4), (VW, VH), fps=fps, codec="libx264", quality=None,
                                             output_params=["-crf", "28", "-preset", "veryfast"], macro_block_size=8)
        writer.send(None)
    tracks, cuts, prev, t_det, t0 = [], [], None, 0.0, time.perf_counter()
    for fid in range(1, N + 1):
        ok, img = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None and float(np.abs(g - prev).mean()) > cut_thresh:
            cuts.append(fid)
        prev = g
        x, r = preprocess(img, cfg)
        a = time.perf_counter()
        pred = det_run(x)
        t_det += time.perf_counter() - a
        cur = {t.track_id: t.tlwh.copy() for t in tr.update(postprocess(pred, r, cfg), img) if keep(t.tlwh, params)}
        tracks.append(cur)
        if writer is not None:
            hud = f"BoT-SORT {tag} | frame {fid}/{N} | tracks {len(cur)}" + (" | SHOT CUT" if cuts and cuts[-1] == fid else "")
            writer.send(np.ascontiguousarray(cv2.resize(draw_tracks(img, cur, hud), (VW, VH))[:, :, ::-1]))
    if writer is not None:
        writer.close()
    wall, n = time.perf_counter() - t0, len(tracks)
    ids = set().union(*[t.keys() for t in tracks]) if tracks else set()
    stats = {"tag": tag, "video": Path(video).name, "fps": fps, "frames": n, "shots": len(cuts) + 1, "unique_ids": len(ids),
             "mean_tracks_per_frame": float(np.mean([len(t) for t in tracks])) if tracks else 0.0,
             "cuts_with_ids_carried_across": sum(1 for c in cuts if 2 <= c <= n and set(tracks[c - 2]) & set(tracks[c - 1])),
             "detector_ms_per_frame": 1000 * t_det / max(n, 1), "reid_ms_per_frame": 1000 * enc.time_s / max(n, 1) if enc else 0.0,
             "reid_batches_per_frame": enc.calls / max(n, 1) if enc else 0.0, "pipeline_fps": n / wall,
             "gmc_failures": getattr(tr.gmc, "failures", 0), "gmc_few_matches_frames": getattr(tr.gmc, "warnings", 0), "mp4": rel(mp4) if write_mp4 else None}
    with open(out_dir / f"{tag}_tracks.json", "w") as f:
        json.dump({"stats": stats, "cuts": cuts, "tracks": [{str(k): [round(float(v), 2) for v in b] for k, b in t.items()} for t in tracks]}, f)
    return {"tracks": tracks, "cuts": cuts, "stats": stats}


def load_video_tracks(path) -> List[Dict[int, np.ndarray]]:
    with open(path) as f:
        return [{int(k): np.array(v) for k, v in t.items()} for t in json.load(f)["tracks"]]


def video_agreement(ref_tracks, test_tracks) -> dict:
    """Score ``test`` tracks against ``ref`` tracks as if ref were ground truth (100 = identical tracking output)."""
    import motmetrics as mm
    import pandas as pd

    def df(tr):
        rows = [[f, tid, *b, 1, -1, -1, -1] for f, t in enumerate(tr, 1) for tid, b in t.items()]
        return pd.DataFrame(rows, columns=["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence", "ClassId", "Visibility", "unused"]).set_index(["FrameId", "Id"])

    acc = mm.utils.compare_to_groundtruth(df(ref_tracks), df(test_tracks), "iou", distth=0.5)
    s = mm.metrics.create().compute(acc, metrics=["idf1", "mota", "num_switches"], name="x")
    return {"IDF1_vs_ref": 100 * float(s["idf1"].iloc[0]), "MOTA_vs_ref": 100 * float(s["mota"].iloc[0]), "IDsw_vs_ref": int(s["num_switches"].iloc[0])}


def video_contact_sheet(mp4, out_jpg, n: int = 6, cols: int = 3, width: int = 1800) -> str:
    """Tile ``n`` evenly spaced frames of an annotated MP4 into one JPEG."""
    mp4 = Path(mp4) if Path(mp4).is_absolute() else NB_DIR / mp4
    cap = cv2.VideoCapture(str(mp4))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for i in np.linspace(0, max(N - 1, 0), n).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            tiles.append(f)
    h, w = tiles[0].shape[:2]
    tw = width // cols
    tiles = [cv2.resize(t, (tw, int(h * tw / w))) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out_jpg), np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(len(tiles) // cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(out_jpg)


def plot_id_timeline(tracks, cuts, out_png, title: str, fps: Optional[float] = None) -> str:
    """One row per track ID (sorted by birth) + red dashed shot cuts: long rows = stable IDs, short rows = ID switches."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = {}
    for f, t in enumerate(tracks, 1):
        for tid in t:
            first.setdefault(tid, f)
    row = {tid: y for y, tid in enumerate(sorted(first, key=first.get))}
    fig, ax = plt.subplots(figsize=(14, max(3, min(12, 0.06 * len(row) + 2))))
    pts = [(f, row[tid]) for f, t in enumerate(tracks, 1) for tid in t]
    if pts:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=1, c=[(y % 20) / 20 for y in ys], cmap="tab20")
    for c in cuts:
        ax.axvline(c, color="red", ls="--", lw=0.6)
    ax.set_xlabel("frame" + (f" ({fps:.0f} fps)" if fps else "") + " - red dashed = detected shot cut")
    ax.set_ylabel("track ID (sorted by birth)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=110)
    plt.close(fig)
    return str(out_png)


def video_preview(mp4, out_mp4, start_s: float = 0, duration_s: float = 10, width: int = 480) -> str:
    """Small H.264 clip cut from an annotated MP4 (to embed in the notebook without bloating it)."""
    import imageio_ffmpeg

    src = Path(mp4) if Path(mp4).is_absolute() else NB_DIR / mp4
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    ok, f = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, f = cap.read()
    h, w = f.shape[:2]
    W, H = width // 8 * 8, int(h * width / w) // 8 * 8
    wr = imageio_ffmpeg.write_frames(str(out_mp4), (W, H), fps=fps, codec="libx264", quality=None,
                                     output_params=["-crf", "30", "-preset", "veryfast"], macro_block_size=8)
    wr.send(None)
    for _ in range(int(duration_s * fps)):
        if not ok:
            break
        wr.send(np.ascontiguousarray(cv2.resize(f, (W, H))[:, :, ::-1]))
        ok, f = cap.read()
    wr.close()
    return str(out_mp4)


# ----------------------------------------------------------------------------------------------------------- benchmarks
def benchmark_torch(model, shape, dtype, device, api: str = "sync", nireq: int = 1, time_s: float = 15.0, name: str = "") -> dict:
    """PyTorch benchmark with ``benchmark_app`` semantics (apples-to-apples with OpenVINO):

    * api sync: 1 request; api async: ``nireq`` requests in flight (own device stream + pinned host buffers + thread each)
    * every iteration: f32 host input -> device (cast to the compute dtype) -> forward -> f32 output back to the host
    * first inference timed separately and excluded; runs for ``time_s`` seconds
    * FPS = batch x iterations / duration; latency per request (AVERAGE is the primary number)
    """
    import statistics

    import torch

    acc = torch.xpu if device.type == "xpu" else (torch.cuda if device.type == "cuda" else None)
    m = model.to(device).to(dtype).eval()
    nireq = 1 if api == "sync" else nireq

    class Req:
        def __init__(self):
            self.stream = acc.Stream() if acc else None
            self.h_in = torch.from_numpy(np.random.rand(*shape).astype(np.float32))
            if acc:
                self.h_in = self.h_in.pin_memory()
            self.d_raw = torch.empty(shape, dtype=torch.float32, device=device)
            self.d_in = torch.empty(shape, dtype=dtype, device=device)
            self.h_out, self.ev = None, acc.Event() if acc else None

        def submit(self):
            ctx = acc.stream(self.stream) if acc else torch.inference_mode()
            with ctx, torch.inference_mode():
                self.d_raw.copy_(self.h_in, non_blocking=True)
                self.d_in.copy_(self.d_raw)
                o = m(self.d_in).float()
                if self.h_out is None:
                    self.h_out = torch.empty(o.shape, dtype=o.dtype, pin_memory=bool(acc))
                self.h_out.copy_(o, non_blocking=True)
                if acc:
                    self.ev.record(self.stream)

        def wait(self):
            if acc:
                self.ev.synchronize()

    reqs = [Req() for _ in range(nireq)]
    t0 = time.perf_counter()
    reqs[0].submit()
    reqs[0].wait()
    first_ms = (time.perf_counter() - t0) * 1000
    for r in reqs:
        r.submit()
        r.wait()
    lat, lock, stop = [], threading.Lock(), [0.0]

    def worker(r):
        loc = []
        while (s := time.perf_counter()) < stop[0]:
            r.submit()
            r.wait()
            loc.append((time.perf_counter() - s) * 1000)
        with lock:
            lat.extend(loc)

    start = time.perf_counter()
    stop[0] = start + time_s
    if nireq == 1:
        worker(reqs[0])
    else:
        th = [threading.Thread(target=worker, args=(r,)) for r in reqs]
        [t.start() for t in th]
        [t.join() for t in th]
    dur = time.perf_counter() - start
    res = {"name": name, "api": api, "nireq": nireq, "iterations": len(lat), "duration_s": dur, "first_inference_ms": first_ms,
           "avg_ms": sum(lat) / len(lat), "median_ms": statistics.median(lat), "min_ms": min(lat), "max_ms": max(lat),
           "fps": shape[0] * len(lat) / dur}
    print(f"[ INFO ] {name} | {api} nireq={nireq} | First inference {first_ms:.1f} ms | Count {len(lat)} iterations | Duration {dur * 1000:.0f} ms")
    print(f"[ INFO ] Latency: Average {res['avg_ms']:.2f} ms  Median {res['median_ms']:.2f}  Min {res['min_ms']:.2f}  Max {res['max_ms']:.2f}")
    print(f"[ INFO ] Throughput: {res['fps']:.2f} FPS")
    return res


def parse_benchmark_log(text: str) -> dict:
    """Parse benchmark_app stdout: AVERAGE / median latency, FPS, api, nireq."""
    g = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, text))  # noqa: E731
    m = re.search(r"Start inference (a?synchronously)(?:, (\d+) inference requests)?", text)
    api = "async" if m and m.group(1) == "asynchronously" else "sync"
    return {"avg_ms": g(r"Average:?\s+([\d.]+)\s*ms"), "median_ms": g(r"Median:?\s+([\d.]+)\s*ms"), "min_ms": g(r"Min:?\s+([\d.]+)\s*ms"),
            "max_ms": g(r"Max:?\s+([\d.]+)\s*ms"), "fps": g(r"Throughput:\s+([\d.]+)\s*FPS"), "iterations": g(r"Count:\s+(\d+)"),
            "api": api, "nireq": int(m.group(2)) if m and m.group(2) else 1}


YoloxDecoded = None
try:
    YoloxDecoded = _decoded_cls()
    YoloxDecoded.__module__ = __name__
except ImportError:  # torch missing (only OpenVINO helpers used)
    pass
setup_environment()
