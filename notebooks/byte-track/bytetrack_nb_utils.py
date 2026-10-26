"""Model conversion, host-side tracking and optional MOT17 evaluation for byte-track.ipynb.

The same preprocessing, postprocessing and BYTETracker run with every
OpenVINO precision. The MOT17 protocol uses the bundled bt_common.py.
"""
import importlib.util
import json
import logging
import os
import re
import sys
import time
import types

import cv2
import numpy as np

from pathlib import Path

# Inputs and generated models are relative to the tutorial directory by default.
NB_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BYTETRACK_PROJECT", NB_DIR)).expanduser().resolve()
REPO = ROOT / "ByteTrack_repo" if (ROOT / "ByteTrack_repo").is_dir() else ROOT / "model" / "ByteTrack_repo"
SCRIPTS = ROOT / "scripts" if (ROOT / "scripts" / "bt_common.py").is_file() else NB_DIR
ARTIFACTS = NB_DIR / "artifacts"                      # hand-over PyTorch notebook -> OpenVINO notebook


def find_file(name, env=None, subdirs=("", "weights", "videos", "data")):
    """First existing <base>/<subdir>/<name> for base in (notebooks/, repo root, repo parent); env var overrides."""
    if env and os.environ.get(env):
        return Path(os.environ[env]).expanduser().resolve()
    for base in (NB_DIR, ROOT, ROOT.parent):
        for sd in subdirs:
            p = base / sd / name
            if p.exists():
                return p.resolve()
    store = ROOT if ROOT != NB_DIR else NB_DIR / "model"
    return store / subdirs[1 if len(subdirs) > 1 else 0] / name


def rel(p):
    """Path relative to the notebooks folder (for printing / storing without machine-specific prefixes)."""
    try:
        return os.path.relpath(str(p), str(NB_DIR))
    except ValueError:
        return str(p)


WEIGHTS = find_file("bytetrack_x_mot17.pth.tar", "BYTETRACK_WEIGHTS")
os.environ.setdefault("BYTETRACK_WEIGHTS", str(WEIGHTS))          # same weights for the scripts/ subprocesses
MOT17_TRAIN = Path(os.environ.get(
    "MOT17_TRAIN",
    ROOT / "datasets" / "MOT17" / "train" if (ROOT / "datasets" / "MOT17" / "train").is_dir()
    else NB_DIR / "data" / "MOT17" / "train",
))
VIDEO = find_file("pedestrians_cc0_30s.mp4", "VIDEO_1")
VIDEO_2 = find_file("Luka_Doncic.mp4", "VIDEO_2")

# ------------------------------------------------------------------------------------------------ environment
for _p in (str(SCRIPTS), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if not hasattr(np, "float"):                         # numpy 2.x shims (ByteTrack matching.py / motmetrics)
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]
if importlib.util.find_spec("cython_bbox") is None:
    def bbox_overlaps(boxes, queries):
        boxes, queries = np.asarray(boxes, np.float64), np.asarray(queries, np.float64)
        if not len(boxes) or not len(queries):
            return np.zeros((len(boxes), len(queries)))
        area_b = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        area_q = (queries[:, 2] - queries[:, 0] + 1) * (queries[:, 3] - queries[:, 1] + 1)
        iw = np.maximum(0, np.minimum(boxes[:, None, 2], queries[None, :, 2]) -
                        np.maximum(boxes[:, None, 0], queries[None, :, 0]) + 1)
        ih = np.maximum(0, np.minimum(boxes[:, None, 3], queries[None, :, 3]) -
                        np.maximum(boxes[:, None, 1], queries[None, :, 1]) + 1)
        intersection = iw * ih
        return intersection / (area_b[:, None] + area_q[None, :] - intersection)

    _bbox_module = types.ModuleType("cython_bbox")
    _bbox_module.bbox_overlaps = bbox_overlaps
    sys.modules["cython_bbox"] = _bbox_module
if importlib.util.find_spec("loguru") is None:
    _m = types.ModuleType("loguru"); _m.logger = logging.getLogger("loguru-stub"); sys.modules["loguru"] = _m
if importlib.util.find_spec("thop") is None:
    _m = types.ModuleType("thop"); _m.profile = lambda *a, **k: (0, 0); sys.modules["thop"] = _m
if importlib.util.find_spec("lap") is None:          # Hungarian solver of BYTETracker -> exact scipy equivalent
    import lap_scipy  # noqa: E402
    sys.modules["lap"] = lap_scipy

import bt_common as BC  # noqa: E402

VIDEO_TRACKER = dict(track_thresh=0.5, track_buffer=30, match_thresh=0.8, min_box_area=10, max_aspect_ratio_w_over_h=1.6, mot20=False)


def load_config(path=None):
    if path is None:
        project_config = ROOT / "openvino_ir" / "bytetrack_tracker_config.json"
        path = project_config if project_config.is_file() else NB_DIR / "bytetrack_tracker_config.json"
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("video_tracker", VIDEO_TRACKER)
    return cfg


# ------------------------------------------------------------------------------------------------ model
def build_yolox_x(weights=WEIGHTS, verbose=True):
    """YOLOX-X exactly as exps/example/mot/yolox_x_mix_det.py (depth 1.33, width 1.25, 1 class)."""
    import torch
    import torch.nn as nn
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

    depth, width, in_ch = 1.33, 1.25, [256, 512, 1024]
    model = YOLOX(YOLOPAFPN(depth, width, in_channels=in_ch), YOLOXHead(1, width, in_channels=in_ch))
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    if verbose:
        print("checkpoint keys:", list(ckpt.keys()), "| tensors:", len(ckpt["model"]))
    model.load_state_dict(ckpt["model"])
    return model.eval()


def _yolox_decoded_cls():
    import torch
    import torch.nn as nn

    class YoloxDecoded(nn.Module):
        """YOLOX + anchor-free grid decode inside the graph -> [B, A, 6] = (cx, cy, w, h, obj, cls), input-pixel space."""

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
            out = self.model(x)
            xy = (out[..., :2] + self.grids) * self.strides
            wh = torch.exp(out[..., 2:4]) * self.strides
            return torch.cat([xy, wh, out[..., 4:]], dim=-1)

    YoloxDecoded.__module__ = __name__
    YoloxDecoded.__qualname__ = "YoloxDecoded"
    return YoloxDecoded


try:
    YoloxDecoded = _yolox_decoded_cls()            # module-level so torch.save / torch.load of the module works
except ImportError:                                # torch not installed (OV-only use of the helpers)
    YoloxDecoded = None


# ------------------------------------------------------------------------------------------------ pre / post
def preprocess(img_bgr, cfg):
    return BC.preprocess(img_bgr, tuple(cfg["detector"]["input_size_hw"]))


def postprocess(pred, cfg):
    """[1, A, 6] -> torch.Tensor [K, 7] (x1, y1, x2, y2, obj, cls, cls_id) in letterboxed-input pixels, or None."""
    pp = cfg["postprocess"]
    return BC.postprocess(np.array(pred, dtype=np.float32, copy=True), pp["conf_thresh"], pp["nms_iou_thresh"])   # yolox postprocess works in place


def draw(img, tracks, hud=None):
    out = img.copy()
    for tid, (x, y, w, h) in tracks.items():
        c = tuple(int(v) for v in np.random.default_rng(int(tid)).integers(60, 255, 3))
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), c, 3)
        cv2.putText(out, str(tid), (int(x), max(20, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
    if hud:
        cv2.putText(out, hud, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
    return out


def draw_dets(img, dets, r, thr=0.5):
    out = img.copy()
    if dets is None:
        return out
    d = dets.cpu().numpy()
    for x1, y1, x2, y2, obj, cls, _ in d:
        if obj * cls < thr:
            continue
        cv2.rectangle(out, (int(x1 / r), int(y1 / r)), (int(x2 / r), int(y2 / r)), (0, 200, 255), 2)
        cv2.putText(out, f"{obj * cls:.2f}", (int(x1 / r), int(y1 / r) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return out


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def score_cosine(a, b):
    """cosine of the detection score maps (obj * cls) - what actually drives NMS / tracking."""
    return cosine(a[..., 4] * a[..., 5], b[..., 4] * b[..., 5])


# ------------------------------------------------------------------------------------------------ runners
def torch_runner(model, device, dtype):
    import torch

    m = model.to(device).to(dtype).eval()

    @torch.inference_mode()
    def run(x):
        return m(torch.from_numpy(x).to(device).to(dtype)).float().cpu().numpy()
    return run


def ov_runner(core, xml_or_model, device="GPU", **props):
    """Sync single-request runner (same as eval_mot17.py: PERFORMANCE_HINT=LATENCY)."""
    cm = core.compile_model(xml_or_model, device, {"PERFORMANCE_HINT": "LATENCY", **props})
    req = cm.create_infer_request()

    def run(x):
        req.infer({0: x})
        return req.get_output_tensor(0).data.copy()
    run.compiled = cm
    run.exec_precision = cm.get_property("INFERENCE_PRECISION_HINT").get_type_name() if device.startswith("GPU") else "n/a"
    return run


def timeit(run, x, n=20, warmup=3):
    for _ in range(warmup):
        y = run(x)
    t = time.perf_counter()
    for _ in range(n):
        y = run(x)
    return y, (time.perf_counter() - t) * 1000 / n


# ------------------------------------------------------------------------------------------------ MOT17 tracking
def mot17_val_half(seqs=None):
    """ByteTrack ablation protocol: second half of each MOT17 train FRCNN sequence. seqs=None -> all 7."""
    return BC.mot17_val_half(str(MOT17_TRAIN), seqs=seqs)


def mot17_train_half_frames():
    import glob
    out = []
    for seq_dir in sorted(glob.glob(f"{MOT17_TRAIN}/MOT17-*-FRCNN")):
        imgs = sorted(glob.glob(f"{seq_dir}/img1/*.jpg"))
        out += imgs[: len(imgs) // 2 + 1]
    return out


def track_sequence(run, seq, cfg, max_frames=None):
    """Identical to scripts/eval_mot17.py. Returns [(fid, tlwhs, ids, scores)], detector ms, total ms."""
    from yolox.tracker.byte_tracker import BYTETracker

    targs = BC.tracker_args(seq["name"], cfg)
    tracker = BYTETracker(targs, frame_rate=cfg["tracker"]["frame_rate"])
    input_hw = tuple(cfg["detector"]["input_size_hw"])
    results, t_det, t_all = [], 0.0, time.perf_counter()
    for fid, path in seq["frames"][:max_frames] if max_frames else seq["frames"]:
        img = cv2.imread(path)
        x, _ = preprocess(img, cfg)
        t0 = time.perf_counter(); pred = run(x); t_det += time.perf_counter() - t0
        dets = postprocess(pred, cfg)
        if dets is None:
            continue
        tlwhs, ids, scores = [], [], []
        for t in tracker.update(dets, img.shape[:2], input_hw):
            tlwh = t.tlwh
            if tlwh[2] * tlwh[3] > targs.min_box_area and tlwh[2] / tlwh[3] <= cfg["tracker"]["max_aspect_ratio_w_over_h"]:
                tlwhs.append(tlwh); ids.append(t.track_id); scores.append(t.score)
        results.append((fid, tlwhs, ids, scores))
    n = len(seq["frames"][:max_frames] if max_frames else seq["frames"])
    return results, 1000 * t_det / n, 1000 * (time.perf_counter() - t_all) / n


def mot_eval(run, cfg, tag, seqs=None, max_frames=None, out_root=f"{NB_DIR}/results/accuracy", verbose=True):
    """Track MOT17 val-half with `run` and score it with motmetrics (same calls as ByteTrack tools/track.py)."""
    import motmetrics as mm

    mm.lap.default_solver = "lap"
    data = mot17_val_half(seqs)
    assert data, f"no MOT17 sequences in {MOT17_TRAIN}"
    res_dir, gt_dir = f"{out_root}/{tag}/track_results", f"{out_root}/_gt_val_half"
    os.makedirs(res_dir, exist_ok=True); os.makedirs(gt_dir, exist_ok=True)
    accs, names, frames, det_ms, tot_ms = [], [], 0, 0.0, 0.0
    for seq in data:
        BC.write_gt_val_half(seq, f"{gt_dir}/{seq['name']}.txt")
        res, d_ms, a_ms = track_sequence(run, seq, cfg, max_frames)
        n = len(seq["frames"][:max_frames] if max_frames else seq["frames"])
        BC.write_results(f"{res_dir}/{seq['name']}.txt", res)
        gt = mm.io.loadtxt(f"{gt_dir}/{seq['name']}.txt", fmt="mot15-2D", min_confidence=1)
        if max_frames:
            gt = gt[gt.index.get_level_values(0) <= max_frames]
        ts = mm.io.loadtxt(f"{res_dir}/{seq['name']}.txt", fmt="mot15-2D", min_confidence=-1)
        accs.append(mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)); names.append(seq["name"])
        frames += n; det_ms += d_ms * n; tot_ms += a_ms * n
        if verbose:
            print(f"  [{tag}] {seq['name']}: {n} frames, detector {d_ms:.1f} ms/frame", flush=True)
    mh = mm.metrics.create()
    s = mh.compute_many(accs, names=names, metrics=["mota", "idf1", "motp", "num_switches", "num_false_positives", "num_misses",
                                                     "mostly_tracked", "mostly_lost"], generate_overall=True)
    o = s.loc["OVERALL"]
    m = {"tag": tag, "frames": frames, "sequences": names, "MOTA": float(o["mota"]) * 100, "IDF1": float(o["idf1"]) * 100,
         "MOTP": float(1 - o["motp"]) * 100, "IDs": int(o["num_switches"]), "FP": int(o["num_false_positives"]),
         "FN": int(o["num_misses"]), "MT": int(o["mostly_tracked"]), "ML": int(o["mostly_lost"]),
         "detector_ms_per_frame": det_ms / frames, "pipeline_ms_per_frame": tot_ms / frames,
         "per_sequence": {k: {"MOTA": float(s.loc[k, "mota"]) * 100, "IDF1": float(s.loc[k, "idf1"]) * 100} for k in names}}
    with open(f"{out_root}/{tag}/metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    return m


# ------------------------------------------------------------------------------------------------ video sanity
def track_video(run, cfg, tag, video=VIDEO, out_dir=f"{NB_DIR}/results/video", max_frames=None, write_mp4=True, cut_thresh=35.0):
    """Track a real video (no GT) with ByteTrack's generic-video parameters; writes an annotated half-res MP4.

    Returns {tracks: [{id: tlwh}], cuts: [frame ids], stats}. Shot cuts = mean |grey diff| of 160x90 thumbnails > cut_thresh.
    """
    from yolox.tracker.basetrack import BaseTrack
    from yolox.tracker.byte_tracker import BYTETracker

    vt = cfg["video_tracker"]
    BaseTrack._count = 0                              # IDs come from a global counter -> reset for comparable runs
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS); W, H = int(cap.get(3)), int(cap.get(4)); N = int(cap.get(7))
    N = min(N, max_frames) if max_frames else N
    tracker = BYTETracker(types.SimpleNamespace(**{k: vt[k] for k in ("track_thresh", "track_buffer", "match_thresh", "mot20")}),
                          frame_rate=round(fps))
    input_hw = tuple(cfg["detector"]["input_size_hw"])
    os.makedirs(out_dir, exist_ok=True)
    out_mp4 = f"{out_dir}/{tag}_tracked.mp4"
    writer = None
    if write_mp4:
        import imageio_ffmpeg
        VW, VH = (W // 2) // 8 * 8, (H // 2) // 8 * 8
        writer = imageio_ffmpeg.write_frames(out_mp4, (VW, VH), fps=fps, codec="libx264", quality=None,
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
        x, _ = preprocess(img, cfg)
        ta = time.perf_counter(); pred = run(x); t_det += time.perf_counter() - ta
        dets = postprocess(pred, cfg)
        online = tracker.update(dets, [H, W], input_hw) if dets is not None else []
        cur = {t.track_id: t.tlwh.copy() for t in online
               if t.tlwh[2] * t.tlwh[3] > vt["min_box_area"] and t.tlwh[2] / t.tlwh[3] <= vt["max_aspect_ratio_w_over_h"]}
        tracks.append(cur)
        if writer is not None:
            hud = f"{tag} | frame {fid}/{N} | tracks {len(cur)}" + (" | SHOT CUT" if cuts and cuts[-1] == fid else "")
            writer.send(np.ascontiguousarray(cv2.resize(draw(img, cur, hud), (VW, VH))[:, :, ::-1]))
    if writer is not None:
        writer.close()
    wall = time.perf_counter() - t0
    n = len(tracks)
    ids = set().union(*[t.keys() for t in tracks]) if tracks else set()
    across = sum(1 for c in cuts if c - 2 >= 0 and c - 1 < n and set(tracks[c - 2]) & set(tracks[c - 1]))
    stats = {"tag": tag, "frames": n, "shots": len(cuts) + 1, "unique_ids": len(ids),
             "mean_tracks_per_frame": float(np.mean([len(t) for t in tracks])) if tracks else 0.0,
             "cuts_with_ids_carried_across": across, "detector_ms_per_frame": 1000 * t_det / max(n, 1),
             "pipeline_fps": n / wall, "mp4": rel(out_mp4) if write_mp4 else None, "video": Path(video).name, "fps": fps}
    with open(f"{out_dir}/{tag}_tracks.json", "w") as f:
        json.dump({"stats": stats, "cuts": cuts, "tracks": [{str(k): [round(float(v), 2) for v in b] for k, b in t.items()} for t in tracks]}, f)
    return {"tracks": tracks, "cuts": cuts, "stats": stats}


def load_video_tracks(path):
    path = Path(path) if Path(path).is_absolute() else NB_DIR / path
    with open(path) as f:
        d = json.load(f)
    return [{int(k): np.array(v) for k, v in t.items()} for t in d["tracks"]]


def video_agreement(ref_tracks, test_tracks):
    """Score `test` against `ref` tracks as if ref were ground truth (100 = identical tracking output)."""
    import motmetrics as mm
    import pandas as pd

    def df(tr):
        rows = [[f, tid, *b, 1, -1, -1, -1] for f, t in enumerate(tr, 1) for tid, b in t.items()]
        return pd.DataFrame(rows, columns=["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence", "ClassId", "Visibility", "unused"]).set_index(["FrameId", "Id"])
    acc = mm.utils.compare_to_groundtruth(df(ref_tracks), df(test_tracks), "iou", distth=0.5)
    s = mm.metrics.create().compute(acc, metrics=["idf1", "mota", "num_switches"], name="x")
    return {"IDF1_vs_ref": float(s["idf1"].iloc[0]) * 100, "MOTA_vs_ref": float(s["mota"].iloc[0]) * 100,
            "IDsw_vs_ref": int(s["num_switches"].iloc[0])}


# ------------------------------------------------------------------------------------------------ benchmark_app
def parse_benchmark_log(path):
    """Parse benchmark_app (or scripts/benchmark_pytorch.py) stdout: average/median latency, FPS, api, nireq."""
    t = open(path).read()
    g = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, t))
    nireq = re.search(r"(\d+) inference requests", t) or re.search(r"nireq=(\d+)", t)
    api = "sync" if re.search(r"inference synchronously|api=sync", t) else "async"
    return {"avg_ms": g(r"Average:?\s+([\d.]+)\s*ms"), "median_ms": g(r"Median:?\s+([\d.]+)\s*ms"),
            "min_ms": g(r"Min:?\s+([\d.]+)\s*ms"), "max_ms": g(r"Max:?\s+([\d.]+)\s*ms"),
            "fps": g(r"Throughput:\s+([\d.]+)\s*FPS"), "iterations": g(r"Count:\s+(\d+)"),
            "nireq": int(nireq.group(1)) if nireq else (1 if api == "sync" else None), "api": api}


def video_contact_sheet(mp4, out_jpg, n=6, cols=3, width=1800):
    """Tile n evenly spaced frames of an (annotated) MP4 into one JPEG for quick viewing in the notebook."""
    mp4 = str(mp4 if Path(mp4).is_absolute() else NB_DIR / mp4)
    out_jpg = str(out_jpg)
    cap = cv2.VideoCapture(mp4)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for i in np.linspace(0, max(N - 1, 0), n).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            tiles.append(f)
    if not tiles:
        return None
    h, w = tiles[0].shape[:2]
    tw = width // cols
    th = int(h * tw / w)
    tiles = [cv2.resize(t, (tw, th)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)])
    cv2.imwrite(out_jpg, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out_jpg


def plot_id_timeline(tracks, cuts, out_png, title, fps=None):
    """One horizontal line per track ID (sorted by first appearance) + red dashed shot cuts: a quick visual check that
    IDs live for whole shots (good) instead of flickering / being reborn (ID switches)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first, last = {}, {}
    for f, t in enumerate(tracks, 1):
        for tid in t:
            first.setdefault(tid, f); last[tid] = f
    ids = sorted(first, key=first.get)
    fig, ax = plt.subplots(figsize=(14, max(3, min(12, 0.06 * len(ids) + 2))))
    for y, tid in enumerate(ids):
        fr = [f for f, t in enumerate(tracks, 1) if tid in t]
        ax.scatter(fr, [y] * len(fr), s=1, color=plt.get_cmap("tab20")(tid % 20))
    for c in cuts:
        ax.axvline(c, color="red", ls="--", lw=0.8)
    ax.set_xlabel("frame" + (f" ({fps:.0f} fps)" if fps else "") + "  -  red dashed = detected shot cut")
    ax.set_ylabel("track ID (sorted by birth)"); ax.set_title(title); ax.set_ylim(-1, len(ids)); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(str(out_png), dpi=110); plt.close(fig)
    return str(out_png)


def video_preview(mp4, out_mp4, start_s=0, duration_s=10, width=480):
    """Cut a small H.264 preview from an annotated MP4 (to embed in the notebook without bloating it)."""
    import imageio_ffmpeg
    src = str(mp4 if Path(mp4).is_absolute() else NB_DIR / mp4)
    out_mp4 = Path(out_mp4)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    ok, f = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, f = cap.read()
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
