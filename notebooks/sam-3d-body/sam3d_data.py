"""Data, metrics and visualization helpers for the SAM 3D Body OpenVINO notebook.

This module is one of three helpers that back ``sam3dbody.ipynb``:

===================  =========================================================
``sam3d_data.py``    sample loading, PCK scoring, rendering       (this file)
``sam3d_torch.py``   PyTorch reference inference + OpenVINO export
``sam3d_ov.py``      OpenVINO IR runtime (pure NumPy + OpenVINO)
===================  =========================================================

**Fully standalone**: NumPy, OpenCV, Matplotlib, pyrender and trimesh only. The
MHR-70 skeleton topology, the skeleton drawing and the mesh renderer are inlined
here, so nothing outside this folder is imported. ``pyrender`` and ``trimesh``
are loaded lazily, so importing this module stays cheap.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# pyrender must pick a headless GL backend *before* it is first imported.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

HERE = Path(__file__).resolve().parent
SAMPLE_DIR = HERE / "sample_data"
DEFAULT_SAMPLE = "000000368212"
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"


# ---------------------------------------------------------------------------
# Keypoint conventions
# ---------------------------------------------------------------------------

#: Index map: COCO-17 joint ``i`` corresponds to MHR-70 joint ``COCO17_TO_MHR70[i]``.
COCO17_TO_MHR70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

NUM_MHR_KEYPOINTS = 70


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as handle:
        handle.write(response.read())


def make_sample(img_bgr: np.ndarray, annotation: dict, image_id=None, file_name=None) -> dict:
    """Bundle an image and a COCO person annotation into the notebook's sample dict.

    Also derives the ``[x1, y1, x2, y2]`` box the model expects and the default
    focal length (the image diagonal, i.e. what the model assumes when no FOV
    estimator is in the loop).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]

    # COCO stores bboxes as [x, y, w, h]; the model expects [x1, y1, x2, y2].
    gx, gy, gw, gh = annotation["bbox"]

    return {
        "image_id": image_id,
        "file_name": file_name,
        "img_bgr": img_bgr,
        "img_rgb": img_rgb,
        "width": img_w,
        "height": img_h,
        "annotation": annotation,
        "bbox_xyxy": np.array([gx, gy, gx + gw, gy + gh], dtype=np.float32),
        "bbox_xywh": np.array([gx, gy, gw, gh], dtype=np.float32),
        "gt_keypoints": np.array(annotation["keypoints"], dtype=np.float32).reshape(17, 3),
        "focal_length": float(np.sqrt(img_h ** 2 + img_w ** 2)),
    }


def load_sample(name: str = DEFAULT_SAMPLE, sample_dir=SAMPLE_DIR, download: bool = True) -> dict:
    """Load the bundled sample image together with its ground-truth annotation.

    ``<sample_dir>/<name>.json`` ships with the notebook and carries the COCO
    annotation. The matching ``.jpg`` is fetched from the COCO servers if it is
    not already on disk.
    """
    sample_dir = Path(sample_dir)
    meta_path = sample_dir / f"{name}.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Sample metadata not found at {meta_path}. It ships with the notebook; "
            "restore it or point `sample_dir` at the folder that contains it."
        )
    meta = json.loads(meta_path.read_text())

    img_path = sample_dir / meta["file_name"]
    if not img_path.exists():
        if not download:
            raise FileNotFoundError(f"Sample image not found at {img_path}")
        _download(meta.get("image_url") or COCO_IMAGE_URL.format(**meta), img_path)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Could not decode {img_path}; delete it and re-run to re-download.")

    sample = make_sample(img_bgr, meta["annotation"], meta.get("image_id"), meta["file_name"])
    sample["path"] = img_path
    sample["source"] = meta.get("source", "")
    return sample


def load_coco_annotations(coco_dir: str):
    """Load full COCO val2017 person-keypoint annotations, for dataset-wide sweeps.

    Returns ``(images, annotations_by_image)``; crowd regions and keypoint-free
    people are dropped.
    """
    ann_path = os.path.join(coco_dir, "annotations", "person_keypoints_val2017.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"COCO annotations not found at {ann_path}.\n"
            "Expected layout: <coco_dir>/val2017/*.jpg and "
            "<coco_dir>/annotations/person_keypoints_val2017.json"
        )
    with open(ann_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations_by_image: dict = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("num_keypoints", 0) > 0 and ann.get("iscrowd", 0) == 0:
            annotations_by_image[ann["image_id"]].append(ann)
    return images, annotations_by_image


# ---------------------------------------------------------------------------
# PCK
# ---------------------------------------------------------------------------

def to_numpy(x) -> np.ndarray:
    """Tensor (any device / framework) or array-like -> NumPy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_pck(pred_kpts, gt_kpts, gt_vis, bbox_xywh, threshold: float = 0.05):
    """Per-keypoint PCK, normalized by the GT bbox diagonal (paper protocol).

    Args:
        pred_kpts: ``[17, 2]`` predicted keypoints, image pixels.
        gt_kpts: ``[17, 2]`` ground-truth keypoints.
        gt_vis: ``[17]`` visibility flags (``> 0`` means visible).
        bbox_xywh: ``[x, y, w, h]`` COCO bbox.
        threshold: fraction of ``sqrt(w^2 + h^2)`` counted as correct.
    Returns:
        ``(correct[17], valid[17])`` boolean arrays.
    """
    _, _, bw, bh = bbox_xywh
    norm = math.sqrt(bw ** 2 + bh ** 2)
    if norm <= 0:
        return np.zeros(17, dtype=bool), np.zeros(17, dtype=bool)
    valid = np.asarray(gt_vis) > 0
    distances = np.linalg.norm(np.asarray(pred_kpts) - np.asarray(gt_kpts), axis=-1)
    correct = (distances / norm < threshold) & valid
    return correct, valid


def compute_person_pck(keypoints_mhr70, annotation, threshold: float = 0.05):
    """PCK@``threshold`` for one person given the model's 70 MHR joints.

    COCO scores 17 joints, so the COCO-17 subset is selected via
    ``COCO17_TO_MHR70`` before the error is normalized by the bbox diagonal.

    Returns ``(pck_percent, correct[17], valid[17])``.
    """
    pred17 = to_numpy(keypoints_mhr70)[COCO17_TO_MHR70]
    gt = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(17, 3)
    correct, valid = compute_pck(pred17, gt[:, :2], gt[:, 2], annotation["bbox"], threshold)
    pck = 100.0 * correct.sum() / max(valid.sum(), 1)
    return float(pck), correct, valid


def pck_tolerance_px(bbox_xywh, threshold: float = 0.05) -> float:
    """Radius, in pixels, inside which a prediction still counts as correct."""
    _, _, bw, bh = bbox_xywh
    return float(threshold * math.sqrt(bw ** 2 + bh ** 2))


# ---------------------------------------------------------------------------
# MHR-70 skeleton topology
# ---------------------------------------------------------------------------

# Palette, then the 65 bones in draw order. Colours are passed straight to
# OpenCV on a BGR image, matching the reference visualizer.
_PALETTE = {
    "T": (51, 153, 255),    # torso / head
    "L": (0, 255, 0),       # left limb + left hand
    "R": (255, 128, 0),     # right limb + right hand
    "P": (255, 153, 255),   # index fingers
    "S": (102, 178, 255),   # middle fingers
    "D": (255, 51, 51),     # ring fingers
}

_BONES = [
    (13, 11, "L"), (11, 9, "L"), (14, 12, "R"), (12, 10, "R"),
    (9, 10, "T"), (5, 9, "T"), (6, 10, "T"), (5, 6, "T"),
    (5, 7, "L"), (6, 8, "R"), (7, 62, "L"), (8, 41, "R"),
    (1, 2, "T"), (0, 1, "T"), (0, 2, "T"), (1, 3, "T"), (2, 4, "T"),
    (3, 5, "T"), (4, 6, "T"),
    (13, 15, "L"), (13, 16, "L"), (13, 17, "L"),
    (14, 18, "R"), (14, 19, "R"), (14, 20, "R"),
    # Left hand: thumb, index, middle, ring, pinky chains hanging off joint 62.
    (62, 45, "R"), (45, 44, "R"), (44, 43, "R"), (43, 42, "R"),
    (62, 49, "P"), (49, 48, "P"), (48, 47, "P"), (47, 46, "P"),
    (62, 53, "S"), (53, 52, "S"), (52, 51, "S"), (51, 50, "S"),
    (62, 57, "D"), (57, 56, "D"), (56, 55, "D"), (55, 54, "D"),
    (62, 61, "L"), (61, 60, "L"), (60, 59, "L"), (59, 58, "L"),
    # Right hand: same five chains hanging off joint 41.
    (41, 24, "R"), (24, 23, "R"), (23, 22, "R"), (22, 21, "R"),
    (41, 28, "P"), (28, 27, "P"), (27, 26, "P"), (26, 25, "P"),
    (41, 32, "S"), (32, 31, "S"), (31, 30, "S"), (30, 29, "S"),
    (41, 36, "D"), (36, 35, "D"), (35, 34, "D"), (34, 33, "D"),
    (41, 40, "L"), (40, 39, "L"), (39, 38, "L"), (38, 37, "L"),
]

#: ``[(joint_a, joint_b, bgr_colour), ...]`` for all 65 MHR-70 bones.
MHR70_SKELETON = [(a, b, _PALETTE[key]) for a, b, key in _BONES]

#: Every MHR-70 keypoint is drawn in the same colour.
MHR70_KEYPOINT_COLOR = _PALETTE["T"]


# ---------------------------------------------------------------------------
# 2D skeleton rendering
# ---------------------------------------------------------------------------

def draw_2d_keypoints(
    img_bgr: np.ndarray,
    keypoints_2d: np.ndarray,
    bbox: Optional[np.ndarray] = None,
    person_idx: int = 0,
    line_width: int = 2,
    radius: int = 5,
) -> np.ndarray:
    """Draw the MHR-70 skeleton (and an optional bbox) on a copy of ``img_bgr``.

    Bones with an endpoint outside the frame are skipped so that off-screen
    joints do not produce long spurious limbs.
    """
    kpts = to_numpy(keypoints_2d)[:NUM_MHR_KEYPOINTS]
    img_vis = img_bgr.copy()
    img_h, img_w = img_vis.shape[:2]

    def inside(pt):
        return 0 < pt[0] < img_w and 0 < pt[1] < img_h

    for a, b, color in MHR70_SKELETON:
        if a >= len(kpts) or b >= len(kpts):
            continue
        pos_a = (int(kpts[a, 0]), int(kpts[a, 1]))
        pos_b = (int(kpts[b, 0]), int(kpts[b, 1]))
        if inside(pos_a) and inside(pos_b):
            cv2.line(img_vis, pos_a, pos_b, color, thickness=line_width)

    for kpt in kpts:
        cv2.circle(img_vis, (int(kpt[0]), int(kpt[1])), radius, MHR70_KEYPOINT_COLOR, -1)

    if bbox is not None:
        x1, y1, x2, y2 = np.asarray(bbox)[:4].astype(int)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_vis, f"P{person_idx}", (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return img_vis


# ---------------------------------------------------------------------------
# 3D mesh rendering
# ---------------------------------------------------------------------------

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def _raymond_lights():
    """Three directional lights in the standard Raymond rig."""
    import pyrender

    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []
    for phi, theta in zip(phis, thetas):
        z = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=matrix
        ))
    return nodes


def render_mesh(
    img_bgr: np.ndarray,
    vertices: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
    faces: np.ndarray,
    mesh_base_color=LIGHT_BLUE,
    scene_bg_color=(1, 1, 1),
    side_view: bool = False,
    rot_angle: float = 90,
) -> np.ndarray:
    """Composite the posed mesh over ``img_bgr`` with a pinhole camera.

    Returns a BGR float image in ``[0, 1]``.
    """
    import pyrender
    import trimesh

    vertices, cam_t, faces = to_numpy(vertices), to_numpy(cam_t), to_numpy(faces)
    image = img_bgr.astype(np.float32) / 255.0
    h, w = image.shape[:2]

    mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
    if side_view:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0]))
    # The model's camera frame is y-down relative to pyrender's.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode="OPAQUE",
        baseColorFactor=(mesh_base_color[2], mesh_base_color[1], mesh_base_color[0], 1.0),
    )
    scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material), "mesh")

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = cam_t * np.array([-1.0, 1.0, 1.0])
    scene.add(
        pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=w / 2.0, cy=h / 2.0, zfar=1e12),
        pose=camera_pose,
    )
    for node in _raymond_lights():
        scene.add_node(node)

    renderer = pyrender.OffscreenRenderer(viewport_height=h, viewport_width=w)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    color = color.astype(np.float32) / 255.0
    alpha = color[:, :, -1][:, :, None]
    return color[:, :, :3] * alpha + (1 - alpha) * image


def render_3d_mesh(img_bgr, vertices, cam_t, focal_length, faces) -> Tuple[np.ndarray, np.ndarray]:
    """Front-view overlay and a side view on white; both ``[H, W, 3]`` BGR uint8."""
    front = render_mesh(img_bgr, vertices, cam_t, focal_length, faces)
    side = render_mesh(
        np.ones_like(img_bgr) * 255, vertices, cam_t, focal_length, faces, side_view=True
    )
    return (front * 255).astype(np.uint8), (side * 255).astype(np.uint8)


def render_views(img_bgr: np.ndarray, person: dict, faces: Optional[np.ndarray]):
    """Render one prediction as ``(skeleton, mesh_front, mesh_side)`` RGB images.

    ``person`` needs ``keypoints_2d`` ``[70, 2]``; the mesh views additionally
    need ``vertices``, ``cam_t`` and ``focal_length`` and are ``None`` otherwise.
    """
    def _rgb(im):
        return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    skeleton = draw_2d_keypoints(img_bgr, person["keypoints_2d"], person.get("bbox"))
    front = side = None
    if person.get("vertices") is not None and faces is not None:
        front, side = render_3d_mesh(
            img_bgr, person["vertices"], person["cam_t"], float(person["focal_length"]), faces
        )
    return _rgb(skeleton), _rgb(front), _rgb(side)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def show_row(
    titled_images: Sequence[Tuple[str, Optional[np.ndarray]]],
    suptitle: Optional[str] = None,
    height: float = 4.4,
):
    """Display a row of ``(title, rgb_image)`` pairs; ``None`` images are skipped."""
    items = [(t, im) for t, im in titled_images if im is not None]
    if not items:
        return
    fig, axes = plt.subplots(1, len(items), figsize=(height * 1.3 * len(items), height))
    for ax, (title, im) in zip(np.atleast_1d(axes), items):
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def free_memory():
    """Release host + accelerator memory between backends (models are multi-GB)."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
