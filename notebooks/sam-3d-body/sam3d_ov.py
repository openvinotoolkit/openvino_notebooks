"""OpenVINO IR runtime for SAM 3D Body — pure NumPy + OpenVINO, no PyTorch.

Companion module to ``sam3dbody.ipynb`` (see ``sam3d_data.py`` for the module
map).

The pipeline is executed as a set of small IRs driven from Python rather than a
single graph, because the decoder runs an *iterative MHR feedback loop* whose
control flow cannot be traced statically:

    backbone (DINOv3 ViT-H/16+)
      -> 6 decoder layers, with MHR feedback after layers 0..4
           (pose head -> MHR mesh head -> 3D joints -> 2D projection ->
            grid_sample of backbone features -> token update)
      -> final pose head -> MHR -> 2D/3D keypoints + mesh vertices

Everything between the IR calls (pose parameter conversion, perspective
projection, cropping) is NumPy, which is where the numbers below come from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import openvino as ov

INPUT_SIZE = 512
BBOX_PADDING = 1.25
NUM_DECODER_LAYERS = 6
NUM_KEYPOINTS = 70
PRECISIONS = ("fp16", "int8")


# ===========================================================================
# Rotation utilities
# ===========================================================================

def rot6d_to_rotmat_np(rot6d: np.ndarray) -> np.ndarray:
    """Convert a 6D rotation representation to a rotation matrix."""
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)  # [..., 3, 3]


def rotmat_to_euler_zyx_np(rotmat: np.ndarray) -> np.ndarray:
    """Rotation matrix -> ZYX Euler angles (matches ``roma.rotmat_to_euler("ZYX")``)."""
    sy = -rotmat[..., 2, 0]
    cy = np.sqrt(rotmat[..., 0, 0] ** 2 + rotmat[..., 1, 0] ** 2)
    y = np.arctan2(sy, cy)
    x = np.arctan2(rotmat[..., 2, 1], rotmat[..., 2, 2])
    z = np.arctan2(rotmat[..., 1, 0], rotmat[..., 0, 0])
    return np.stack([z, y, x], axis=-1)


def batch_xyz_from_6d_np(rot6d: np.ndarray) -> np.ndarray:
    """Batch of 6D rotations -> XYZ Euler angles (matches ``batchXYZfrom6D``)."""
    x_raw = rot6d[..., :3]
    y_raw = rot6d[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw, axis=-1)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x, axis=-1)
    matrix = np.stack([x, y, z], axis=-1)  # [..., 3, 3]
    sy = np.sqrt(matrix[..., 0, 0] ** 2 + matrix[..., 1, 0] ** 2)
    singular = (sy < 1e-6).astype(np.float32)
    ex = np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2])
    ey = np.arctan2(-matrix[..., 2, 0], sy)
    ez = np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
    exs = np.arctan2(-matrix[..., 1, 2], matrix[..., 1, 1])
    eys = np.arctan2(-matrix[..., 2, 0], sy)
    ezs = np.zeros_like(ez)
    return np.stack([
        ex * (1 - singular) + exs * singular,
        ey * (1 - singular) + eys * singular,
        ez * (1 - singular) + ezs * singular,
    ], axis=-1)


# ===========================================================================
# Pose parameter conversion (519 raw decoder dims -> MHR inputs)
# ===========================================================================

class PoseParamConverter:
    """Converts the raw 519-dim decoder output into MHR-ready parameters.

    Loaded once from ``pose_head_buffers.npz``; all math is pure NumPy.
    """

    def __init__(self, buffers_path: str):
        data = np.load(buffers_path, allow_pickle=True)
        self.scale_mean = data["scale_mean"]                        # [68]
        self.scale_comps = data["scale_comps"]                      # [28, 68]
        self.hand_pose_mean = data["hand_pose_mean"]                # [54]
        self.hand_pose_comps = data["hand_pose_comps"]              # [54, 54]
        self.hand_joint_idxs_left = data["hand_joint_idxs_left"]    # [27]
        self.hand_joint_idxs_right = data["hand_joint_idxs_right"]  # [27]
        self.keypoint_mapping = data["keypoint_mapping"]            # [308, 18566]
        self.mhr_param_hand_mask = data["mhr_param_hand_mask"]      # [133] bool
        self.body_3dof_rot_idxs = data["body_3dof_rot_idxs"]        # [23, 3]
        self.body_1dof_rot_idxs = data["body_1dof_rot_idxs"]        # [58]
        self.body_1dof_trans_idxs = data["body_1dof_trans_idxs"]    # [6]
        self.hand_mask_cont_3dof = data["hand_mask_cont_3dof"]      # [54] bool
        self.hand_mask_cont_1dof = data["hand_mask_cont_1dof"]      # [54] bool
        self.hand_mask_model_3dof = data["hand_mask_model_3dof"]    # [27] bool
        self.hand_mask_model_1dof = data["hand_mask_model_1dof"]    # [27] bool
        self.faces = data["faces"] if "faces" in data else None

    def compact_cont_to_model_params_body(self, body_cont: np.ndarray) -> np.ndarray:
        """Continuous body pose (260) -> Euler model params (133)."""
        batch_size = body_cont.shape[0]
        num_3dof = 23 * 3  # 69
        num_1dof = 58
        cont_3dofs = body_cont[:, :2 * num_3dof].reshape(batch_size, -1, 6)
        cont_1dofs = body_cont[:, 2 * num_3dof:2 * num_3dof + 2 * num_1dof].reshape(batch_size, -1, 2)
        cont_trans = body_cont[:, 2 * num_3dof + 2 * num_1dof:]
        params_3dofs = batch_xyz_from_6d_np(cont_3dofs).reshape(batch_size, -1)
        params_1dofs = np.arctan2(cont_1dofs[:, :, 0], cont_1dofs[:, :, 1])
        body_params = np.zeros((batch_size, 133), dtype=body_cont.dtype)
        body_params[:, self.body_3dof_rot_idxs.flatten()] = params_3dofs
        body_params[:, self.body_1dof_rot_idxs] = params_1dofs
        body_params[:, self.body_1dof_trans_idxs] = cont_trans
        return body_params

    def compact_cont_to_model_params_hand(self, hand_cont: np.ndarray) -> np.ndarray:
        """Continuous hand pose (54) -> Euler model params (27)."""
        batch_size = hand_cont.shape[0]
        cont_3dofs = hand_cont[:, self.hand_mask_cont_3dof].reshape(batch_size, -1, 6)
        cont_1dofs = hand_cont[:, self.hand_mask_cont_1dof].reshape(batch_size, -1, 2)
        params_3dofs = batch_xyz_from_6d_np(cont_3dofs).reshape(batch_size, -1)
        params_1dofs = np.arctan2(cont_1dofs[:, :, 0], cont_1dofs[:, :, 1])
        hand_params = np.zeros((batch_size, 27), dtype=hand_cont.dtype)
        hand_params[:, self.hand_mask_model_3dof] = params_3dofs
        hand_params[:, self.hand_mask_model_1dof] = params_1dofs
        return hand_params

    def convert(self, pose_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw 519-dim decoder output -> ``(shape[B,45], model_params[B,204], face[B,72])``."""
        batch_size = pose_params.shape[0]
        count = 0
        global_rot_6d = pose_params[:, count:count + 6]; count += 6
        body_cont = pose_params[:, count:count + 260]; count += 260
        shape_params = pose_params[:, count:count + 45]; count += 45
        scale_params = pose_params[:, count:count + 28]; count += 28
        hand_params_raw = pose_params[:, count:count + 108]; count += 108
        face_params = pose_params[:, count:count + 72]
        face_expr = face_params * 0.0
        global_rot_euler = rotmat_to_euler_zyx_np(rot6d_to_rotmat_np(global_rot_6d))
        global_trans = np.zeros((batch_size, 3), dtype=pose_params.dtype)
        body_pose_euler = self.compact_cont_to_model_params_body(body_cont)
        body_pose_euler[:, self.mhr_param_hand_mask] = 0.0
        body_pose_euler[:, -3:] = 0.0
        left_hand_cont = self.hand_pose_mean[None, :] + hand_params_raw[:, :54] @ self.hand_pose_comps
        right_hand_cont = self.hand_pose_mean[None, :] + hand_params_raw[:, 54:] @ self.hand_pose_comps
        left_hand_euler = self.compact_cont_to_model_params_hand(left_hand_cont)
        right_hand_euler = self.compact_cont_to_model_params_hand(right_hand_cont)
        full_pose = np.concatenate([global_trans * 10, global_rot_euler, body_pose_euler[:, :130]], axis=-1)
        full_pose[:, self.hand_joint_idxs_left] = left_hand_euler
        full_pose[:, self.hand_joint_idxs_right] = right_hand_euler
        scales = self.scale_mean[None, :] + scale_params @ self.scale_comps
        model_params = np.concatenate([full_pose, scales], axis=-1)
        return shape_params, model_params, face_expr

    def get_3d_keypoints(self, verts: np.ndarray, skel_state: np.ndarray) -> np.ndarray:
        """MHR output -> ``[B, 70, 3]`` joints in meters, camera coordinate system."""
        joint_coords = skel_state[:, :, :3]
        model_vert_joints = np.concatenate([verts / 100.0, joint_coords / 100.0], axis=1)
        j3d = np.einsum("kv,bvc->bkc", self.keypoint_mapping, model_vert_joints)[:, :NUM_KEYPOINTS, :]
        j3d[:, :, 1] *= -1
        j3d[:, :, 2] *= -1
        return j3d


# ===========================================================================
# Geometry
# ===========================================================================

def camera_project_2d(
    j3d: np.ndarray,
    camera_params: np.ndarray,
    bbox_center: np.ndarray,
    bbox_scale_w: np.ndarray,
    img_w: int,
    img_h: int,
    focal_length: float,
) -> np.ndarray:
    """Full-perspective 3D -> 2D projection (matches the PyTorch ``PerspectiveHead``).

    Args:
        j3d: ``[B, N, 3]`` 3D keypoints from MHR.
        camera_params: ``[B, 3]`` raw ``(s, tx, ty)`` from the decoder.
        bbox_center: ``[B, 2]`` bbox center in original-image pixels.
        bbox_scale_w: ``[B]`` bbox scale width.
        img_w, img_h: original image size.
        focal_length: focal length in pixels.
    Returns:
        ``[B, N, 2]`` keypoints in original-image pixel coordinates.
    """
    cam_t = camera_translation(camera_params, bbox_center, bbox_scale_w, img_w, img_h, focal_length)
    j3d_cam = j3d + cam_t[:, None, :]
    z = j3d_cam[:, :, 2:3]
    kps_2d_x = focal_length * j3d_cam[:, :, 0:1] / z + img_w / 2.0
    kps_2d_y = focal_length * j3d_cam[:, :, 1:2] / z + img_h / 2.0
    return np.concatenate([kps_2d_x, kps_2d_y], axis=-1)


def camera_translation(
    camera_params: np.ndarray,
    bbox_center: np.ndarray,
    bbox_scale_w: np.ndarray,
    img_w: int,
    img_h: int,
    focal_length: float,
) -> np.ndarray:
    """Decoder camera params ``(s, tx, ty)`` -> full-frame translation ``[B, 3]``."""
    s = -camera_params[:, 0]
    tx = camera_params[:, 1]
    ty = -camera_params[:, 2]
    bs = bbox_scale_w * s + 1e-8
    tz = 2.0 * focal_length / bs
    cx_offset = 2.0 * (bbox_center[:, 0] - img_w / 2.0) / bs
    cy_offset = 2.0 * (bbox_center[:, 1] - img_h / 2.0) / bs
    return np.stack([tx + cx_offset, ty + cy_offset, tz], axis=-1)


def _bbox_to_square_scale(bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bbox ``[x1,y1,x2,y2]`` -> ``(center[2], square_scale[2])`` (GetBBoxCenterScale)."""
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    scale = np.array([(x2 - x1) * BBOX_PADDING, (y2 - y1) * BBOX_PADDING], dtype=np.float32)
    # Fix aspect ratio: first to 0.75 (w/h), then square.
    w_s, h_s = scale[0], scale[1]
    if w_s > h_s * 0.75:
        scale = np.array([w_s, w_s / 0.75], dtype=np.float32)
    else:
        scale = np.array([h_s * 0.75, h_s], dtype=np.float32)
    max_dim = max(scale[0], scale[1])
    return center, np.array([max_dim, max_dim], dtype=np.float32)


def _affine_warp_matrix(center: np.ndarray, scale: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
    """Affine matrix mapping the padded bbox onto an ``input_size`` square (TopdownAffine)."""
    src_dir = np.array([0.0, scale[0] * -0.5], dtype=np.float32)
    dst_dir = np.array([0.0, float(input_size) * -0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = np.array([src[0, 0] - (src[1, 1] - src[0, 1]),
                       src[0, 1] + (src[1, 0] - src[0, 0])], dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [input_size * 0.5, input_size * 0.5]
    dst[1] = dst[0] + dst_dir
    dst[2] = np.array([dst[0, 0] - (dst[1, 1] - dst[0, 1]),
                       dst[0, 1] + (dst[1, 0] - dst[0, 0])], dtype=np.float32)
    return cv2.getAffineTransform(src, dst)


def preprocess_image(
    img_rgb: np.ndarray,
    bbox: np.ndarray,
    input_size: int = INPUT_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.float32, np.ndarray, np.ndarray]:
    """Crop and condition one person, matching ``GetBBoxCenterScale`` + ``TopdownAffine``.

    Returns ``(img_tensor[1,3,S,S] in [0,1], condition_info[1,3], bbox_center[2],
    bbox_scale_w, ray_cond[1,2,S,S], warp_mat[2,3])``. ImageNet normalization is
    baked into the exported backbone IR, so it is deliberately not applied here.
    ``bbox_scale_w`` stays ``float32`` so the projection math below runs in the
    same precision as the reference pipeline.
    """
    img_h, img_w = img_rgb.shape[:2]
    bbox_center, bbox_scale = _bbox_to_square_scale(np.asarray(bbox, dtype=np.float32))
    warp_mat = _affine_warp_matrix(bbox_center, bbox_scale, input_size)

    cropped = cv2.warpAffine(img_rgb, warp_mat, (input_size, input_size), flags=cv2.INTER_LINEAR)
    img_tensor = (cropped.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    # CLIFF-style bbox conditioning.
    focal_length = np.sqrt(float(img_h ** 2 + img_w ** 2))
    condition_info = np.array([[
        (bbox_center[0] - img_w / 2.0) / focal_length,
        (bbox_center[1] - img_h / 2.0) / focal_length,
        bbox_scale[0] / focal_length,
    ]], dtype=np.float32)

    # Ray conditioning: per-pixel camera ray directions in original image space.
    grid_x, grid_y = np.meshgrid(
        np.arange(input_size, dtype=np.float32), np.arange(input_size, dtype=np.float32)
    )
    ax, ay = warp_mat[0, 0], warp_mat[1, 1]
    tx_w, ty_w = warp_mat[0, 2], warp_mat[1, 2]
    orig_x = grid_x / ax - tx_w / ax
    orig_y = grid_y / ay - ty_w / ay
    ray_x = (orig_x - img_w / 2.0) / focal_length
    ray_y = (orig_y - img_h / 2.0) / focal_length
    ray_cond = np.stack([ray_x, ray_y], axis=0)[None].astype(np.float32)

    return img_tensor, condition_info, bbox_center, bbox_scale[0], ray_cond, warp_mat


# ===========================================================================
# OpenVINO plumbing
# ===========================================================================

# Maps a requested precision to the OpenVINO device execution-precision hint.
# This must be set EXPLICITLY: OpenVINO does not derive execution precision from
# the IR's element types, and every device has a different default
# (CPU=f32, GPU=f16, NPU=f16). INT8 weight compression is graph-driven, so the
# hint only governs the surrounding floating-point activations.
_PREC_TO_HINT = {"fp16": "f16", "int8": "f16"}


def apply_precision_hint(core: ov.Core, device: str, precision: str, tag: str = "OV") -> Optional[str]:
    """Set ``INFERENCE_PRECISION_HINT`` on ``core`` for ``device``; returns the hint."""
    hint = _PREC_TO_HINT.get(str(precision).lower())
    if hint is None:
        print(f"[{tag}] Unknown precision '{precision}'; leaving the device default unchanged.")
        return None
    try:
        core.set_property(device, {"INFERENCE_PRECISION_HINT": hint})
        print(f"[{tag}] {device} INFERENCE_PRECISION_HINT = {hint}  (precision={precision})")
    except Exception as exc:
        print(f"[{tag}] Could not set {device} precision hint '{hint}': {exc}")
    return hint


def _infer(model, inputs: dict) -> dict:
    """Run a compiled OV model. Copies are cheap next to the compute time."""
    return model(inputs, share_inputs=False, share_outputs=False)


def _out(model, result, index: int = 0) -> np.ndarray:
    return np.array(result[model.output(index)], copy=True)


def bilinear_grid_sample(features: np.ndarray, grid: np.ndarray, ov_model=None) -> np.ndarray:
    """Bilinear sampling of ``features[B,C,H,W]`` at ``grid[B,N,1,2]`` in ``[-1, 1]``.

    Uses the exported OpenVINO ``GridSample`` op when available (the GPU plugin
    lowers it to an optimized OpenCL kernel), otherwise a NumPy fallback.
    Returns ``[B, C, N, 1]``.
    """
    if ov_model is not None:
        result = ov_model({0: features, 1: grid}, share_inputs=False, share_outputs=False)
        return np.array(result[ov_model.output(0)], copy=True)

    features = np.ascontiguousarray(features)
    B, C, H, W = features.shape
    N = grid.shape[1]

    gx, gy = grid[:, :, 0, 0], grid[:, :, 0, 1]
    px = np.clip(((gx + 1.0) * W - 1.0) / 2.0, 0, W - 1.001)
    py = np.clip(((gy + 1.0) * H - 1.0) / 2.0, 0, H - 1.001)

    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    wx = px - x0.astype(np.float32)
    wy = py - y0.astype(np.float32)

    output = np.zeros((B, C, N), dtype=np.float32)
    for b in range(B):
        feat_flat = features[b].reshape(C, H * W)
        f00 = feat_flat[:, y0[b] * W + x0[b]]
        f01 = feat_flat[:, y0[b] * W + x1[b]]
        f10 = feat_flat[:, y1[b] * W + x0[b]]
        f11 = feat_flat[:, y1[b] * W + x1[b]]
        w00 = ((1.0 - wx[b]) * (1.0 - wy[b]))[None, :]
        w01 = (wx[b] * (1.0 - wy[b]))[None, :]
        w10 = ((1.0 - wx[b]) * wy[b])[None, :]
        w11 = (wx[b] * wy[b])[None, :]
        output[b] = f00 * w00 + f01 * w01 + f10 * w10 + f11 * w11

    return output[:, :, :, None]


# ===========================================================================
# Inference pipeline
# ===========================================================================

class Sam3DBodyOpenVINO:
    """SAM 3D Body inference on OpenVINO IR, with the full MHR feedback loop.

    Mirrors the PyTorch reference architecture exactly: 6 decoder layers, MHR
    feedback after layers 0-4, token update from the predicted 2D/3D keypoints,
    and a full-perspective camera projection.
    """

    #: Files that must exist under ``model_dir`` for the pipeline to load.
    REQUIRED = (
        "backbone/backbone_{precision}.xml",
        "mhr/mhr_{precision}.xml",
        "iterative/ray_cond_emb.xml",
        "iterative/decoder_layer_5.xml",
        "iterative/grid_sample.xml",
        "iterative/buffers/iterative_buffers.npz",
        "iterative/buffers/pose_head_buffers.npz",
    )

    @classmethod
    def is_available(cls, model_dir, precision: str) -> bool:
        """True when every IR/buffer this class needs is present in ``model_dir``."""
        model_dir = Path(model_dir)
        return all((model_dir / f.format(precision=precision)).exists() for f in cls.REQUIRED)

    def __init__(
        self,
        model_dir,
        device: str = "CPU",
        precision: str = "fp16",
        cache_dir: Optional[str] = None,
    ):
        """Compile every IR under ``model_dir`` onto ``device``.

        Args:
            model_dir: directory containing ``backbone/``, ``iterative/`` and ``mhr/``.
            device: OpenVINO device (``CPU``, ``GPU``, ``AUTO``, ...).
            precision: ``fp16`` or ``int8``. Selects the IR and, crucially, the
                device execution-precision hint.
            cache_dir: on-disk kernel cache (defaults to ``<model_dir>/cache``).
        """
        model_dir = Path(model_dir)
        iter_dir = model_dir / "iterative"
        self.model_dir = model_dir
        self.device = device
        self.precision = str(precision).lower()

        backbone_ir = self._find_ir(model_dir / "backbone", "backbone_*.xml")
        mhr_ir = self._find_ir(model_dir / "mhr", "mhr_*.xml")
        pose_buffers_path = iter_dir / "buffers" / "pose_head_buffers.npz"
        if not pose_buffers_path.exists():
            raise FileNotFoundError(
                f"pose_head_buffers.npz not found at {pose_buffers_path}\n"
                f"Export the IR first: python sam3d_torch.py --precision {self.precision}"
            )

        self.core = ov.Core()
        # Persistent kernel cache: the first run JIT-compiles GPU kernels, later
        # runs load them from disk, removing the cold-start latency that would
        # otherwise pollute the first person's timing.
        cache_dir = str(cache_dir or (model_dir / "cache"))
        os.makedirs(cache_dir, exist_ok=True)
        self.core.set_property({"CACHE_DIR": cache_dir})
        self.inference_precision = apply_precision_hint(
            self.core, device, self.precision, tag="Sam3DBodyOV"
        )

        def compile_ir(path, required: bool = True):
            path = Path(path)
            if not path.exists():
                if required:
                    raise FileNotFoundError(f"Missing OpenVINO IR: {path}")
                return None
            return self.core.compile_model(self.core.read_model(str(path)), device)

        print(f"[Sam3DBodyOV] Loading backbone: {backbone_ir.name}")
        self.backbone = compile_ir(backbone_ir)
        print(f"[Sam3DBodyOV] Loading MHR: {mhr_ir.name}")
        self.mhr = compile_ir(mhr_ir)

        print(f"[Sam3DBodyOV] Loading {NUM_DECODER_LAYERS} decoder layers and heads...")
        self.ray_cond_emb = compile_ir(iter_dir / "ray_cond_emb.xml")
        self.decoder_layers = [
            compile_ir(iter_dir / f"decoder_layer_{i}.xml") for i in range(NUM_DECODER_LAYERS)
        ]
        self.decoder_norm = compile_ir(iter_dir / "decoder_norm.xml")
        self.pose_proj = compile_ir(iter_dir / "pose_proj.xml")
        self.camera_proj = compile_ir(iter_dir / "camera_proj.xml")
        self.kp_posemb_2d = compile_ir(iter_dir / "kp_posemb_2d.xml")
        self.kp_feat_linear = compile_ir(iter_dir / "kp_feat_linear.xml")
        self.kp_posemb_3d = compile_ir(iter_dir / "kp_posemb_3d.xml")
        self.init_to_token = compile_ir(iter_dir / "init_to_token.xml")
        self.prev_to_token = compile_ir(iter_dir / "prev_to_token.xml")
        self.grid_sample = compile_ir(iter_dir / "grid_sample.xml", required=False)
        if self.grid_sample is None:
            print("[Sam3DBodyOV] grid_sample.xml not found; using the NumPy fallback")

        print("[Sam3DBodyOV] Loading buffers...")
        buf = np.load(str(iter_dir / "buffers" / "iterative_buffers.npz"))
        self.init_pose = buf["init_pose"].astype(np.float32)
        self.init_camera = buf["init_camera"].astype(np.float32)
        self.prompt_embed = buf["prompt_embed"].astype(np.float32)
        self.image_pe = buf["image_pe"].astype(np.float32)
        self.kp_embed = buf["keypoint_embedding"].astype(np.float32)
        self.kp3d_embed = buf["keypoint3d_embedding"].astype(np.float32)

        self.converter = PoseParamConverter(str(pose_buffers_path))
        self.faces = self.converter.faces
        self.pelvis_idx = [9, 10]
        print(f"[Sam3DBodyOV] Ready. device={device} precision={self.precision}")

    @staticmethod
    def _find_ir(directory: Path, pattern: str) -> Path:
        matches = sorted(directory.glob(pattern)) if directory.exists() else []
        if not matches:
            raise FileNotFoundError(f"No IR matching '{pattern}' under {directory}")
        return matches[0]

    # -- warmup ------------------------------------------------------------

    def warmup(self, n: int = 2) -> None:
        """Run dummy inferences so kernels are JIT-compiled and cached on disk.

        Without this the first real person's latency includes the full GPU
        kernel-compilation cost (hundreds of ms), which inflates and
        destabilises the reported timings.
        """
        dummy_img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        dummy_bbox = np.array([0.0, 0.0, float(INPUT_SIZE), float(INPUT_SIZE)], dtype=np.float32)
        for _ in range(max(1, n)):
            try:
                self.infer_single(
                    dummy_img, dummy_bbox,
                    focal_length=float(np.sqrt(2.0) * INPUT_SIZE),
                )
            except Exception as exc:
                print(f"[Sam3DBodyOV] warmup skipped ({exc})")
                break

    # -- pipeline stages ---------------------------------------------------

    def _init_tokens(self, condition_info: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build the initial decoder tokens and their positional augment."""
        init_input = np.concatenate([condition_info, self.init_pose, self.init_camera], axis=-1).astype(np.float32)
        prev_input = np.concatenate([self.init_pose, self.init_camera], axis=-1).astype(np.float32)

        tok0 = _out(self.init_to_token, _infer(self.init_to_token, {0: init_input}))
        tok1 = _out(self.prev_to_token, _infer(self.prev_to_token, {0: prev_input}))
        tok2 = self.prompt_embed

        tokens = np.concatenate([
            tok0[:, None, :], tok1[:, None, :], tok2[:, None, :],
            self.kp_embed[None, :, :], self.kp3d_embed[None, :, :],
        ], axis=1).astype(np.float32)

        zeros_1 = np.zeros((1, 1, 1024), dtype=np.float32)
        zeros_kp = np.zeros((1, NUM_KEYPOINTS, 1024), dtype=np.float32)
        augment = np.concatenate([
            zeros_1, tok1[:, None, :], tok2[:, None, :], zeros_kp, zeros_kp,
        ], axis=1).astype(np.float32)

        return tokens, augment

    def _run_pose_head(self, tokens_normed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pose and camera heads on the normalized tokens (residual on the init pose)."""
        pose_token = tokens_normed[:, 0, :]
        pose_delta = _out(self.pose_proj, _infer(self.pose_proj, {0: pose_token}))
        cam_delta = _out(self.camera_proj, _infer(self.camera_proj, {0: pose_token}))
        return pose_delta + self.init_pose, cam_delta + self.init_camera

    def _run_mhr(self, pose_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """MHR mesh head -> ``(j3d[B,70,3], verts[B,V,3])`` (vertices in cm)."""
        shape, model_params, face_expr = self.converter.convert(pose_params)
        result = _infer(self.mhr, {
            "identity_coeffs": shape,
            "model_params": model_params,
            "face_expr_coeffs": face_expr,
        })
        verts = _out(self.mhr, result, 0)
        skel_state = _out(self.mhr, result, 1)
        return self.converter.get_3d_keypoints(verts, skel_state), verts

    @staticmethod
    def _full_to_crop(j2d: np.ndarray, warp_mat: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
        """Full-image 2D keypoints -> crop coordinates in ``[-0.5, 0.5]``."""
        B, N, _ = j2d.shape
        j2d_homo = np.concatenate([j2d, np.ones((B, N, 1), dtype=j2d.dtype)], axis=-1)
        return np.einsum("ij,bnj->bni", warp_mat, j2d_homo) / float(input_size) - 0.5

    @staticmethod
    def _compute_2d_depth(j3d, camera_params, bbox_scale_w, focal_length) -> np.ndarray:
        """Per-keypoint camera-space depth, used to mask out keypoints behind the camera."""
        s = -camera_params[:, 0]
        tz = 2.0 * focal_length / (bbox_scale_w * s + 1e-8)
        return j3d[:, :, 2] + tz[:, None]

    def _token_update(self, tokens, augment, j2d_crop, j3d, depth, features):
        """Feed the current MHR prediction back into the keypoint tokens."""
        KPS_START, KPS3D_START = 3, 3 + NUM_KEYPOINTS

        tokens = tokens.copy()
        augment = augment.copy()

        kps_01 = j2d_crop + 0.5
        invalid = (
            (kps_01[:, :, 0] < 0) | (kps_01[:, :, 0] > 1)
            | (kps_01[:, :, 1] < 0) | (kps_01[:, :, 1] > 1)
            | (depth < 1e-5)
        )
        valid_f = (~invalid[:, :, None]).astype(np.float32)

        posemb_2d = _out(self.kp_posemb_2d, _infer(self.kp_posemb_2d, {0: j2d_crop.astype(np.float32)}))
        augment[:, KPS_START:KPS_START + NUM_KEYPOINTS, :] = posemb_2d * valid_f

        grid = (j2d_crop * 2.0)[:, :, None, :].astype(np.float32)
        sampled = bilinear_grid_sample(features, grid, ov_model=self.grid_sample)
        sampled = sampled.squeeze(3).transpose(0, 2, 1) * valid_f

        feat_embed = _out(self.kp_feat_linear, _infer(self.kp_feat_linear, {0: sampled.astype(np.float32)}))
        tokens[:, KPS_START:KPS_START + NUM_KEYPOINTS, :] += feat_embed

        pelvis_center = (j3d[:, [self.pelvis_idx[0]], :] + j3d[:, [self.pelvis_idx[1]], :]) / 2.0
        posemb_3d = _out(
            self.kp_posemb_3d, _infer(self.kp_posemb_3d, {0: (j3d - pelvis_center).astype(np.float32)})
        )
        augment[:, KPS3D_START:KPS3D_START + NUM_KEYPOINTS, :] = posemb_3d

        return tokens, augment

    # -- public API --------------------------------------------------------

    def infer_single(
        self,
        img_rgb: np.ndarray,
        bbox: np.ndarray,
        focal_length: Optional[float] = None,
        mask_embedding: Optional[np.ndarray] = None,
    ) -> dict:
        """Run the full iterative pipeline for one person.

        Args:
            img_rgb: ``[H, W, 3]`` uint8 RGB image.
            bbox: ``[4]`` person box ``[x1, y1, x2, y2]``.
            focal_length: defaults to the image diagonal.
            mask_embedding: optional ``[1, 1280, 32, 32]`` mask conditioning.
        Returns:
            dict with ``j3d`` ``[70,3]``, ``j2d`` ``[70,2]``, ``verts`` ``[V,3]``,
            ``camera_params`` ``[3]``, ``cam_t`` ``[3]`` and ``focal_length``.
        """
        img_h, img_w = img_rgb.shape[:2]
        if focal_length is None:
            focal_length = float(np.sqrt(img_h ** 2 + img_w ** 2))

        img_tensor, condition_info, bbox_center, bbox_scale_w, ray_cond, warp_mat = preprocess_image(
            img_rgb, bbox
        )

        features = _out(self.backbone, _infer(self.backbone, {0: img_tensor}))

        if self.ray_cond_emb is not None:
            features = _out(self.ray_cond_emb, _infer(self.ray_cond_emb, {
                0: features.astype(np.float32), 1: ray_cond.astype(np.float32),
            }))
        if mask_embedding is not None:
            features = features + mask_embedding.astype(np.float32)

        B, C, H, W = features.shape
        context = features.reshape(B, C, H * W).transpose(0, 2, 1).copy()
        image_augment = self.image_pe
        tokens, augment = self._init_tokens(condition_info)

        project = lambda j3d, cam: camera_project_2d(
            j3d, cam, bbox_center=bbox_center[None], bbox_scale_w=np.array([bbox_scale_w]),
            img_w=img_w, img_h=img_h, focal_length=focal_length,
        )

        for layer_idx in range(NUM_DECODER_LAYERS):
            layer = self.decoder_layers[layer_idx]
            result = _infer(layer, {
                0: tokens.astype(np.float32),
                1: context.astype(np.float32),
                2: augment.astype(np.float32),
                3: image_augment.astype(np.float32),
            })
            tokens = _out(layer, result, 0)
            context = _out(layer, result, 1)

            if layer_idx == NUM_DECODER_LAYERS - 1:
                break

            # MHR feedback: predict, project, and re-inject into the tokens.
            tokens_normed = _out(self.decoder_norm, _infer(self.decoder_norm, {0: tokens.astype(np.float32)}))
            pose_params, camera_params = self._run_pose_head(tokens_normed)
            j3d, _ = self._run_mhr(pose_params)
            j2d_crop = self._full_to_crop(project(j3d, camera_params), warp_mat)
            depth = self._compute_2d_depth(j3d, camera_params, bbox_scale_w, focal_length)
            tokens, augment = self._token_update(tokens, augment, j2d_crop, j3d, depth, features)

        tokens_normed = _out(self.decoder_norm, _infer(self.decoder_norm, {0: tokens.astype(np.float32)}))
        pose_params, camera_params = self._run_pose_head(tokens_normed)
        j3d, verts = self._run_mhr(pose_params)
        j2d = project(j3d, camera_params)
        cam_t = camera_translation(
            camera_params, bbox_center[None], np.array([bbox_scale_w]), img_w, img_h, focal_length
        )

        # MHR emits centimeters in a y/z-flipped frame; PyTorch reports meters.
        vis_verts = verts / 100.0
        vis_verts[..., [1, 2]] *= -1

        return {
            "j3d": j3d[0],
            "j2d": j2d[0],
            "verts": vis_verts[0],
            "camera_params": camera_params[0],
            "cam_t": cam_t[0],
            "focal_length": focal_length,
        }
