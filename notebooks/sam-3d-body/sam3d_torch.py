"""PyTorch side of the SAM 3D Body OpenVINO notebook: reference inference + export.

Companion module to ``sam3dbody.ipynb`` (see ``sam3d_data.py`` for the module
map). Two responsibilities:

1. :class:`Sam3DBodyTorch` — runs the reference checkpoint on Intel XPU when one
   is available, and on CPU otherwise.
2. :func:`export_openvino_ir` — traces every sub-module and writes the
   OpenVINO IR consumed by ``sam3d_ov.py``, plus :class:`DenseMHR`, a dense
   re-implementation of the MHR head that *is* exportable.

Both need the ``sam_3d_body`` package \u2014 the network definition the checkpoint's
weights are loaded into. A copy ships in this folder, so nothing outside it is
ever imported and the whole directory can be moved elsewhere as a unit.
``sam3d_data.py`` and ``sam3d_ov.py`` do not touch it at all, so the OpenVINO
half runs without the model source.

Inference and export target different devices, so the device shims are applied
on demand rather than at import time. Run the export in a **separate process**:

    python sam3d_torch.py --precision fp16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import openvino as ov

try:
    import nncf
    HAS_NNCF = True
except ImportError:  # INT8 export only
    HAS_NNCF = False

PRECISIONS = ("fp16", "int8")

#: Every artifact lives beside these helpers, so this folder can be copied
#: anywhere and still run without reaching outside itself.
HERE = Path(__file__).resolve().parent

CHECKPOINT_DIR = HERE / "checkpoints" / "sam-3d-body-dinov3"
DEFAULT_CHECKPOINT = str(CHECKPOINT_DIR / "model.ckpt")
DEFAULT_MHR_PATH = str(CHECKPOINT_DIR / "assets" / "mhr_model.pt")
DEFAULT_OUTPUT_DIR = str(HERE / "ov_models")


def model_source_root(start=None) -> Path:
    """Locate the ``sam_3d_body`` package and make it importable.

    Prefers the copy vendored next to these helpers so the folder stays
    portable, then falls back to a parent checkout. ``SAM3D_MODEL_SRC``
    overrides both.
    """
    env_root = os.environ.get("SAM3D_MODEL_SRC")
    if env_root:
        candidates = [Path(env_root).resolve()]
    else:
        start = Path(start).resolve() if start else HERE
        candidates = [start, *start.parents]

    for candidate in candidates:
        if (candidate / "sam_3d_body" / "__init__.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    raise RuntimeError(
        "Could not locate the 'sam_3d_body' package. It normally ships beside these "
        "helpers; set SAM3D_MODEL_SRC to point at a checkout instead."
    )


#: Kept so notebooks written against the earlier name keep working.
find_repo_root = model_source_root

MODEL_SRC = model_source_root()


# ===========================================================================
# Device redirection
# ===========================================================================
# Only two devices are ever used here: Intel XPU when present, CPU otherwise.
# The model source, which must stay unmodified, reaches for a vendor-specific
# accelerator in two places -- torch's legacy per-vendor tensor transfer method,
# and `recursive_to(batch, <vendor>)`. The shims below retarget both at the
# selected device, so no vendor runtime is ever required or initialised.

#: Destinations the model asks for when it wants results back on the host.
_HOST_TARGETS = ("cpu", "numpy")

#: torch's legacy per-vendor transfer method, rebound to plain ``Tensor.to()``.
_LEGACY_TRANSFER_METHOD = "cuda"

_ORIGINALS: dict = {}
_PATCH_TARGET: Optional[str] = None


def enable_device_patches(device: Union[str, torch.device]) -> None:
    """Retarget the model's hard-coded accelerator calls at ``device``."""
    global _PATCH_TARGET
    target = torch.device(device).type
    if _PATCH_TARGET == target:
        return

    model_source_root()
    import sam_3d_body.models.meta_arch.sam3d_body as sam3d_mod
    import sam_3d_body.sam_3d_body_estimator as estimator_mod
    import sam_3d_body.utils as sam_utils

    if not _ORIGINALS:
        _ORIGINALS["recursive_to"] = sam_utils.recursive_to
        _ORIGINALS["jit_load"] = torch.jit.load

    orig_recursive_to = _ORIGINALS["recursive_to"]
    orig_jit_load = _ORIGINALS["jit_load"]

    def recursive_to_device(data, requested):
        return orig_recursive_to(data, requested if requested in _HOST_TARGETS else target)

    # Patch the definition and the two modules that imported it by value.
    sam_utils.recursive_to = recursive_to_device
    estimator_mod.recursive_to = recursive_to_device
    sam3d_mod.recursive_to = recursive_to_device

    setattr(torch.Tensor, _LEGACY_TRANSFER_METHOD,
            lambda self, *args, **kwargs: self.to(target))

    def jit_load_on_host(f, map_location=None, **kwargs):
        # TorchScript sub-models load on the host; load_sam_3d_body moves them after.
        return orig_jit_load(f, map_location="cpu", **kwargs)

    torch.jit.load = jit_load_on_host
    _PATCH_TARGET = target


def xpu_available() -> bool:
    """True when this PyTorch build can reach an Intel XPU."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def resolve_device(device: Union[str, torch.device] = "cpu") -> torch.device:
    """Validate a requested device, falling back to CPU when XPU is unavailable."""
    device = torch.device(device)
    if device.type not in ("cpu", "xpu"):
        raise ValueError(f"Unsupported device '{device}'; expected 'cpu' or 'xpu'.")
    if device.type == "xpu" and not xpu_available():
        print("[Sam3DBodyTorch] XPU requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return device


# ===========================================================================
# Reference inference
# ===========================================================================

class Sam3DBodyTorch:
    """Reference SAM 3D Body inference on Intel XPU or CPU.

    The detector, segmentor and FOV estimator of the full pipeline are not
    instantiated: the notebook supplies a ground-truth box and the default
    (image-diagonal) focal length, which isolates the pose model itself.

    Args:
        checkpoint_path: path to ``model.ckpt``.
        mhr_path: path to ``mhr_model.pt``.
        device: ``"cpu"`` (default, works everywhere) or ``"xpu"`` for an Intel
            GPU, which is roughly 8x faster but needs a torch+xpu build.
            An unavailable XPU falls back to CPU.
    """

    def __init__(
        self,
        checkpoint_path: str,
        mhr_path: str = "",
        device: Union[str, torch.device] = "cpu",
    ):
        self.device = resolve_device(device)
        enable_device_patches(self.device)

        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

        print(f"[Sam3DBodyTorch] Device: {self.device}")
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path, device=self.device, mhr_path=mhr_path
        )
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.model,
            model_cfg=self.model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        #: ``[F, 3]`` mesh topology, shared by every renderer.
        self.faces = self.estimator.faces
        self.last_inference_time_ms: float = 0.0
        self._inference_times: List[float] = []

    def infer(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        bbox_thr: float = 0.5,
        use_mask: bool = False,
    ) -> List[dict]:
        """Run inference on one image; returns one prediction dict per person.

        Args:
            img: image path, or an already-loaded RGB array.
            bboxes: ``(N, 4)`` person boxes ``[x1, y1, x2, y2]``.
            masks: optional pre-computed masks.
            cam_int: optional camera intrinsics.
            bbox_thr: detection confidence threshold (unused without a detector).
            use_mask: enable mask-conditioned inference.
        """
        if self.device.type == "xpu":
            torch.xpu.synchronize()
        t0 = time.perf_counter()

        results = self.estimator.process_one_image(
            img, bboxes=bboxes, masks=masks, cam_int=cam_int,
            bbox_thr=bbox_thr, use_mask=use_mask,
        )

        if self.device.type == "xpu":
            torch.xpu.synchronize()
        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0
        self._inference_times.append(self.last_inference_time_ms)
        return results

    def get_timing_stats(self) -> dict:
        """Aggregated wall-clock statistics over every :meth:`infer` call."""
        if not self._inference_times:
            return {}
        times = np.array(self._inference_times)
        return {
            "num_calls": len(times),
            "total_ms": float(times.sum()),
            "mean_ms": float(times.mean()),
            "median_ms": float(np.median(times)),
            "std_ms": float(times.std()),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
        }

    def reset_timing(self) -> None:
        self._inference_times.clear()
        self.last_inference_time_ms = 0.0


# ===========================================================================
# Dense MHR — an exportable re-implementation of the MHR TorchScript head
# ===========================================================================
# The shipped MHR model uses aten::sparse_coo_tensor, prim::Loop (FK traversal)
# and prim::unchecked_cast, none of which the OpenVINO frontend supports. The
# math below is identical, expressed with dense ops only.
# Quaternion convention: XYZW (x, y, z, w), matching pymomentum.

def euler_xyz_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    """XYZ Euler angles ``[..., 3]`` (radians) -> quaternion ``[..., 4]`` XYZW."""
    half = euler * 0.5
    cx, sx = torch.cos(half[..., 0:1]), torch.sin(half[..., 0:1])
    cy, sy = torch.cos(half[..., 1:2]), torch.sin(half[..., 1:2])
    cz, sz = torch.cos(half[..., 2:3]), torch.sin(half[..., 2:3])

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return torch.cat([x, y, z, w], dim=-1)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two XYZW quaternions."""
    x1, y1, z1, w1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    x2, y2, z2, w2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.cat([x, y, z, w], dim=-1)


def quaternion_rotate_point(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotate a 3D point ``[..., 3]`` by an XYZW unit quaternion."""
    p_quat = torch.cat([p, torch.zeros_like(p[..., :1])], dim=-1)
    q_conj = torch.cat([-q[..., 0:3], q[..., 3:4]], dim=-1)
    return quaternion_multiply(quaternion_multiply(q, p_quat), q_conj)[..., 0:3]


def batch_6d_from_xyz(euler_xyz: torch.Tensor) -> torch.Tensor:
    """XYZ Euler angles -> the first two rotation-matrix columns, flattened ``[..., 6]``."""
    cx, sx = torch.cos(euler_xyz[..., 0]), torch.sin(euler_xyz[..., 0])
    cy, sy = torch.cos(euler_xyz[..., 1]), torch.sin(euler_xyz[..., 1])
    cz, sz = torch.cos(euler_xyz[..., 2]), torch.sin(euler_xyz[..., 2])

    # R = Rz @ Ry @ Rx (extrinsic XYZ)
    r00 = cy * cz
    r01 = sx * sy * cz - cx * sz
    r10 = cy * sz
    r11 = sx * sy * sz + cx * cz
    r20 = -sy
    r21 = sx * cy
    return torch.stack([r00, r10, r20, r01, r11, r21], dim=-1)


class DenseMHR(nn.Module):
    """Dense, OpenVINO-exportable re-implementation of the pymomentum MHR model."""

    N_VERTS = 18439
    N_JOINTS = 127

    def __init__(self):
        super().__init__()
        # Populated by from_torchscript().
        self.register_buffer("base_shape", torch.zeros(self.N_VERTS, 3))
        self.register_buffer("shape_vectors", torch.zeros(45, self.N_VERTS, 3))
        self.register_buffer("face_shape_vectors", torch.zeros(72, self.N_VERTS, 3))

        self.register_buffer("joint_translation_offsets", torch.zeros(self.N_JOINTS, 3))
        self.register_buffer("joint_prerotations", torch.zeros(self.N_JOINTS, 4))
        self.register_buffer("joint_parents", torch.zeros(self.N_JOINTS, dtype=torch.long))

        self.register_buffer("parameter_transform", torch.zeros(889, 249))
        self.register_buffer("pose_parameters_mask", torch.zeros(249, dtype=torch.bool))
        self.register_buffer("scaling_parameters_mask", torch.zeros(249, dtype=torch.bool))

        self.register_buffer("inverse_bind_pose", torch.zeros(self.N_JOINTS, 8))
        self.register_buffer("skinning_weights", torch.zeros(self.N_VERTS, self.N_JOINTS))

        # Dense replacement for the SparseLinear pose-corrective predictor.
        self.register_buffer("pose_correctives_weight", torch.zeros(3000, 750))
        self.pose_correctives_dense = nn.Linear(3000, 55317, bias=False)

        self.register_buffer("minmax_min", torch.zeros(198))
        self.register_buffer("minmax_max", torch.zeros(198))
        self.register_buffer("minmax_parameter_index", torch.zeros(198, dtype=torch.long))

    @classmethod
    def from_torchscript(cls, mhr_path: str, device: str = "cpu") -> "DenseMHR":
        """Build from the shipped TorchScript MHR file, densifying sparse tensors."""
        mhr = torch.jit.load(mhr_path, map_location=device)
        mhr.eval()

        model = cls()
        bufs = dict(mhr.named_buffers())
        params = dict(mhr.named_parameters())

        model.base_shape = bufs["character_torch.blend_shape.base_shape"].clone()
        model.shape_vectors = bufs["character_torch.blend_shape.shape_vectors"].clone()
        model.face_shape_vectors = bufs["face_expressions_model.shape_vectors"].clone()

        model.joint_translation_offsets = bufs["character_torch.skeleton.joint_translation_offsets"].clone()
        model.joint_prerotations = bufs["character_torch.skeleton.joint_prerotations"].clone()
        model.joint_parents = bufs["character_torch.skeleton.joint_parents"].clone().long()

        model.parameter_transform = bufs["character_torch.parameter_transform.parameter_transform"].clone()
        model.pose_parameters_mask = bufs["character_torch.parameter_transform.pose_parameters"].clone()
        model.scaling_parameters_mask = bufs["character_torch.parameter_transform.scaling_parameters"].clone()

        # Linear blend skinning: scatter the flattened sparse weights into a dense matrix.
        skin_indices = bufs["character_torch.linear_blend_skinning.skin_indices_flattened"].long()
        skin_weights = bufs["character_torch.linear_blend_skinning.skin_weights_flattened"]
        vert_indices = bufs["character_torch.linear_blend_skinning.vert_indices_flattened"].long()
        model.inverse_bind_pose = bufs["character_torch.linear_blend_skinning.inverse_bind_pose"].clone()

        skinning_weights = torch.zeros(cls.N_VERTS, cls.N_JOINTS)
        valid = skin_indices >= 0
        skinning_weights.index_put_(
            (vert_indices[valid], skin_indices[valid]), skin_weights[valid], accumulate=True
        )
        model.skinning_weights = skinning_weights

        # Pose correctives: COO -> dense [3000, 750].
        sparse_indices = params["pose_correctives_model.pose_dirs_predictor.0.sparse_indices"].data
        sparse_weight = params["pose_correctives_model.pose_dirs_predictor.0.sparse_weight"].data
        dense_weight = torch.zeros(3000, 750)
        dense_weight[sparse_indices[0].long(), sparse_indices[1].long()] = sparse_weight
        model.pose_correctives_weight = dense_weight

        model.pose_correctives_dense.weight = nn.Parameter(
            params["pose_correctives_model.pose_dirs_predictor.2.weight"].data.clone(),
            requires_grad=False,
        )

        model.minmax_min = bufs["character_torch.parameter_limits.minmax_min"].clone()
        model.minmax_max = bufs["character_torch.parameter_limits.minmax_max"].clone()
        model.minmax_parameter_index = bufs["character_torch.parameter_limits.minmax_parameter_index"].clone().long()

        model.eval()
        return model

    def blend_shapes(self, identity_coeffs: torch.Tensor) -> torch.Tensor:
        """Identity coefficients ``[B, 45]`` -> rest-pose vertices ``[B, V, 3]``."""
        offsets = torch.einsum("nvd,...n->...vd", self.shape_vectors, identity_coeffs)
        return offsets + self.base_shape

    def face_expressions(self, expr_coeffs: torch.Tensor) -> torch.Tensor:
        """Expression coefficients ``[B, 72]`` -> vertex offsets ``[B, V, 3]``."""
        return torch.einsum("nvd,...n->...vd", self.face_shape_vectors, expr_coeffs)

    def model_parameters_to_joint_parameters(self, model_params: torch.Tensor) -> torch.Tensor:
        """Model params ``[B, 204]`` -> joint params ``[B, 889]`` (127 joints x 7).

        The TorchScript graph pads the input to 249 with zeros before the
        transform, so the trailing 45 columns can simply be dropped.
        """
        return torch.mm(model_params, self.parameter_transform[:, :204].T)

    def apply_parameter_limits(self, joint_params: torch.Tensor) -> torch.Tensor:
        """Clamp joint parameters to their valid ranges."""
        if self.minmax_min.shape[0] > 0:
            idx = self.minmax_parameter_index.long()
            joint_params = joint_params.clone()
            joint_params[:, idx] = torch.clamp(joint_params[:, idx], self.minmax_min, self.minmax_max)
        return joint_params

    def joint_parameters_to_local_state(self, joint_params: torch.Tensor) -> torch.Tensor:
        """Joint params ``[B, 889]`` -> local state ``[B, 127, 8]`` = pos, quat XYZW, scale."""
        batch_size = joint_params.shape[0]
        jp = joint_params.reshape(batch_size, self.N_JOINTS, 7)

        trans = jp[..., :3] + self.joint_translation_offsets.unsqueeze(0)
        local_quat = euler_xyz_to_quaternion(jp[..., 3:6])
        prerot = self.joint_prerotations.unsqueeze(0).expand(batch_size, -1, -1)
        local_quat = quaternion_multiply(prerot, local_quat)
        scale = torch.exp(jp[..., 6:7] * 0.6931471824645996)  # log2 -> linear scale

        return torch.cat([trans, local_quat, scale], dim=-1)

    def forward_kinematics(self, local_state: torch.Tensor) -> torch.Tensor:
        """Local -> global skeleton state ``[B, 127, 8]`` by walking the joint tree.

        Natural index order is already topological (parent index < child index).
        """
        global_state = torch.zeros_like(local_state)
        global_state[:, 0] = local_state[:, 0]  # root has no parent

        for j in range(1, self.N_JOINTS):
            parent = self.joint_parents[j].item()
            p_pos = global_state[:, parent, :3]
            p_quat = global_state[:, parent, 3:7]
            p_scale = global_state[:, parent, 7:8]

            l_pos = local_state[:, j, :3]
            l_quat = local_state[:, j, 3:7]
            l_scale = local_state[:, j, 7:8]

            global_state[:, j, :3] = p_pos + quaternion_rotate_point(p_quat, l_pos * p_scale)
            global_state[:, j, 3:7] = quaternion_multiply(p_quat, l_quat)
            global_state[:, j, 7:8] = p_scale * l_scale

        return global_state

    def linear_blend_skinning(self, rest_verts: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """Skin ``[B, V, 3]`` rest vertices with the dense skinning weight matrix."""
        batch_size, n_verts = rest_verts.shape[0], rest_verts.shape[1]

        inv_pos = self.inverse_bind_pose[:, :3]
        inv_quat = self.inverse_bind_pose[:, 3:7]
        inv_scale = self.inverse_bind_pose[:, 7:8]

        g_pos = global_state[:, :, :3]
        g_quat = global_state[:, :, 3:7]
        g_scale = global_state[:, :, 7:8]

        # Compose "undo bind pose" with "apply current pose" into one transform.
        combined_quat = quaternion_multiply(g_quat, inv_quat.unsqueeze(0).expand(batch_size, -1, -1))
        combined_scale = g_scale * inv_scale.unsqueeze(0)
        combined_pos = g_pos + quaternion_rotate_point(
            g_quat, inv_pos.unsqueeze(0).expand(batch_size, -1, -1) * g_scale
        )

        x, y = combined_quat[..., 0:1], combined_quat[..., 1:2]
        z, w = combined_quat[..., 2:3], combined_quat[..., 3:4]
        rot = torch.zeros(batch_size, self.N_JOINTS, 3, 3, device=rest_verts.device)
        rot[..., 0, 0] = (1 - 2 * (y * y + z * z)).squeeze(-1)
        rot[..., 0, 1] = (2 * (x * y - w * z)).squeeze(-1)
        rot[..., 0, 2] = (2 * (x * z + w * y)).squeeze(-1)
        rot[..., 1, 0] = (2 * (x * y + w * z)).squeeze(-1)
        rot[..., 1, 1] = (1 - 2 * (x * x + z * z)).squeeze(-1)
        rot[..., 1, 2] = (2 * (y * z - w * x)).squeeze(-1)
        rot[..., 2, 0] = (2 * (x * z - w * y)).squeeze(-1)
        rot[..., 2, 1] = (2 * (y * z + w * x)).squeeze(-1)
        rot[..., 2, 2] = (1 - 2 * (x * x + y * y)).squeeze(-1)
        rot = rot * combined_scale.unsqueeze(-1)

        # Blend the per-joint [R|t] matrices per vertex, then apply them.
        transform = torch.cat([rot, combined_pos.unsqueeze(-1)], dim=-1)
        blended = torch.einsum("vj,bjk->bvk", self.skinning_weights, transform.reshape(batch_size, self.N_JOINTS, 12))
        blended_transform = blended.reshape(batch_size, n_verts, 3, 4)

        rest_homo = torch.cat([rest_verts, torch.ones_like(rest_verts[..., :1])], dim=-1)
        return torch.einsum("bvij,bvj->bvi", blended_transform, rest_homo)

    def pose_correctives(self, joint_params: torch.Tensor) -> torch.Tensor:
        """Pose-dependent vertex corrections ``[B, V, 3]`` (SparseLinear -> ReLU -> Linear)."""
        batch_size = joint_params.shape[0]

        jp = joint_params.reshape(batch_size, -1, 7)
        euler = jp[:, 2:, 3:6]  # skip root and pelvis
        rot_6d = batch_6d_from_xyz(euler)
        # Center on the identity rotation [1, 0, 0, 0, 1, 0].
        rot_6d[:, :, 0] = rot_6d[:, :, 0] - 1.0
        rot_6d[:, :, 4] = rot_6d[:, :, 4] - 1.0

        x = torch.mm(rot_6d.reshape(batch_size, -1), self.pose_correctives_weight.T)
        x = self.pose_correctives_dense(F.relu(x))
        return x.reshape(batch_size, -1, 3)

    def forward(
        self,
        identity_coeffs: torch.Tensor,
        model_params: torch.Tensor,
        face_expr_coeffs: torch.Tensor,
    ) -> tuple:
        """``([B,45], [B,204], [B,72])`` -> ``(skinned_verts[B,V,3], skel_state[B,127,8])``."""
        rest_verts = self.blend_shapes(identity_coeffs)
        joint_params = self.model_parameters_to_joint_parameters(model_params)
        # Parameter limits are intentionally not applied: the TorchScript model skips them.
        global_state = self.forward_kinematics(self.joint_parameters_to_local_state(joint_params))

        unposed_verts = (
            rest_verts
            + self.face_expressions(face_expr_coeffs)
            + self.pose_correctives(joint_params)
        )
        return self.linear_blend_skinning(unposed_verts, global_state), global_state

    def verify(self, mhr_ts, n_samples: int = 5, atol: float = 1e-3) -> bool:
        """Check numerical equivalence against the original TorchScript model."""
        all_pass = True
        for i in range(n_samples):
            shape = torch.randn(1, 45)
            model = torch.randn(1, 204) * 0.1
            expr = torch.randn(1, 72) * 0.1
            with torch.no_grad():
                out_ts = mhr_ts(shape, model, expr)
                out_dense = self(shape, model, expr)
            diff_v = (out_ts[0] - out_dense[0]).abs().max().item()
            diff_s = (out_ts[1] - out_dense[1]).abs().max().item()
            ok = diff_v < atol and diff_s < atol
            all_pass &= ok
            print(f"  Sample {i}: verts_diff={diff_v:.6e}, skel_diff={diff_s:.6e} "
                  f"[{'PASS' if ok else 'FAIL'}]")
        return all_pass


# ===========================================================================
# Export wrappers
# ===========================================================================

class BackboneWrapper(nn.Module):
    """Backbone with ImageNet normalization baked in.

    Inference preprocessing only scales images to ``[0, 1]``; folding mean/std
    in here reproduces the model's ``data_preprocess()`` inside the IR.
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.register_buffer("image_mean", model.image_mean.clone())
        self.register_buffer("image_std", model.image_std.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.image_mean) / self.image_std
        backbone_dtype = next(self.backbone.encoder.parameters()).dtype
        features = self.backbone(x.to(backbone_dtype))
        if isinstance(features, tuple):
            features = features[-1]
        return features.float()


class RayCondEmbWrapper(nn.Module):
    def __init__(self, ray_cond_emb):
        super().__init__()
        self.ray_cond_emb = ray_cond_emb

    def forward(self, image_embeddings, ray_cond):
        return self.ray_cond_emb(image_embeddings, ray_cond)


class DecoderLayerWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, tokens, context, token_pe, context_pe):
        tokens_out, context_out = self.layer(tokens, context, token_pe, context_pe, None)
        return tokens_out, context_out


class ModuleWrapper(nn.Module):
    """Trace a single-input sub-module (norm / linear / MLP) as its own IR."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


class GridSampleModel(nn.Module):
    """``F.grid_sample`` as a standalone op, matching the inference call exactly.

    Inputs ``features[B,C,H,W]`` and ``grid[B,N,1,2]``; output ``[B,C,N,1]``.
    Carries no weights, so its ``.bin`` is empty by design.
    """

    def forward(self, features, grid):
        return F.grid_sample(features, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class CameraProjection(nn.Module):
    """Perspective 3D->2D projection; mirrors ``camera_project_2d`` in ``sam3d_ov.py``."""

    def forward(self, j3d, camera_params, bbox_center, bbox_scale_w, img_size, focal_length):
        s = -camera_params[:, 0]
        tx = camera_params[:, 1]
        ty = -camera_params[:, 2]
        img_w, img_h = img_size[:, 0], img_size[:, 1]
        bs = bbox_scale_w * s + 1e-8
        tz = 2.0 * focal_length / bs
        cx_offset = 2.0 * (bbox_center[:, 0] - img_w / 2.0) / bs
        cy_offset = 2.0 * (bbox_center[:, 1] - img_h / 2.0) / bs
        cam_t = torch.stack([tx + cx_offset, ty + cy_offset, tz], dim=-1)
        j3d_cam = j3d + cam_t[:, None, :]
        z = j3d_cam[:, :, 2:3]
        kps_2d_x = focal_length[:, None, None] * j3d_cam[:, :, 0:1] / z + img_w[:, None, None] / 2.0
        kps_2d_y = focal_length[:, None, None] * j3d_cam[:, :, 1:2] / z + img_h[:, None, None] / 2.0
        return torch.cat([kps_2d_x, kps_2d_y], dim=-1)


class FullToCrop(nn.Module):
    """Full-image 2D keypoints -> crop-space coordinates in ``[-0.5, 0.5]``."""

    def forward(self, j2d, warp_mat):
        B, N, _ = j2d.shape
        j2d_homo = torch.cat([j2d, torch.ones(B, N, 1, dtype=j2d.dtype, device=j2d.device)], dim=-1)
        return torch.einsum("ij,bnj->bni", warp_mat, j2d_homo) / 512.0 - 0.5


# ===========================================================================
# Export helpers
# ===========================================================================

def save_ov_model(ov_model, output_path: str, precision: str) -> float:
    """Write an OV model at ``precision``; returns the ``.bin`` size in MB.

    ``int8`` uses NNCF *weight-only* compression: no calibration data is
    required and activations stay floating point.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if precision == "fp16":
        ov.save_model(ov_model, output_path, compress_to_fp16=True)
    elif precision == "int8":
        if not HAS_NNCF:
            raise RuntimeError("NNCF is required for INT8 export. Run: pip install nncf")
        ov.save_model(nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM), output_path)
    else:
        raise ValueError(f"Unknown precision '{precision}'; expected one of {PRECISIONS}")

    return os.path.getsize(output_path.replace(".xml", ".bin")) / (1024 * 1024)


def trace_and_convert(module: nn.Module, example_inputs):
    """``torch.jit.trace`` a module and convert the trace to an OpenVINO model."""
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs)
    return ov.convert_model(traced, example_input=example_inputs)


def _export(module: nn.Module, example_inputs, output_dir, name: str, precision: str) -> float:
    out_path = os.path.join(str(output_dir), f"{name}.xml")
    size = save_ov_model(trace_and_convert(module.float(), example_inputs), out_path, precision)
    print(f"    - {name}.xml ({size:.2f} MB)")
    return size


def load_reference_model(checkpoint_path: str, mhr_path: str):
    """Load the full SAM 3D Body model on CPU, ready for tracing."""
    enable_device_patches("cpu")
    from sam_3d_body.build_models import load_sam_3d_body

    model, _ = load_sam_3d_body(checkpoint_path=checkpoint_path, device="cpu", mhr_path=mhr_path)
    return model.eval().float()


def export_backbone(model, output_dir, precision: str) -> float:
    """DINOv3 ViT-H/16+ backbone, ``[1,3,512,512] -> [1,1280,32,32]``."""
    print(f"\n  [Backbone] {precision.upper()}")
    return _export(BackboneWrapper(model), (torch.randn(1, 3, 512, 512),),
                   output_dir, f"backbone_{precision}", precision)


def export_mhr(mhr_path: str, output_dir, precision: str) -> float:
    """Dense MHR mesh + skeleton head."""
    print(f"\n  [MHR] {precision.upper()}")
    dense_mhr = DenseMHR.from_torchscript(mhr_path, device="cpu").float()
    example = (torch.randn(1, 45), torch.randn(1, 204), torch.randn(1, 72))
    return _export(dense_mhr, example, output_dir, f"mhr_{precision}", precision)


def export_mask_encoder(model, output_dir, precision: str) -> float:
    """Mask conditioning CNN, ``[1,1,512,512] -> [1,1280,32,32]``, plus its no-mask buffer."""
    print(f"\n  [MaskEncoder] {precision.upper()}")
    size = _export(model.prompt_encoder.mask_downscaling, (torch.randn(1, 1, 512, 512),),
                   output_dir, f"mask_encoder_{precision}", precision)
    os.makedirs(str(output_dir), exist_ok=True)
    np.savez(
        os.path.join(str(output_dir), "mask_buffers.npz"),
        no_mask_embed=model.prompt_encoder.no_mask_embed.weight.data.cpu().numpy(),
    )
    print("    - mask_buffers.npz")
    return size


def export_iterative_components(model, output_dir, precision: str) -> dict:
    """The 6 decoder layers plus the norm, pose/camera heads and token MLPs."""
    print(f"\n  [Iterative decoder] {precision.upper()}")
    B, N_TOKENS, C_DECODER, HW, C_BACKBONE = 1, 143, 1024, 1024, 1280
    sizes = {}

    sizes["ray_cond_emb"] = _export(
        RayCondEmbWrapper(model.ray_cond_emb),
        (torch.randn(B, C_BACKBONE, 32, 32), torch.randn(B, 2, 512, 512)),
        output_dir, "ray_cond_emb", precision,
    )

    layer_example = (
        torch.randn(B, N_TOKENS, C_DECODER), torch.randn(B, HW, C_BACKBONE),
        torch.randn(B, N_TOKENS, C_DECODER), torch.randn(B, HW, C_BACKBONE),
    )
    for i, layer in enumerate(model.decoder.layers):
        sizes[f"decoder_layer_{i}"] = _export(
            DecoderLayerWrapper(layer), layer_example, output_dir, f"decoder_layer_{i}", precision
        )

    sizes["decoder_norm"] = _export(
        ModuleWrapper(model.decoder.norm_final), (torch.randn(B, N_TOKENS, C_DECODER),),
        output_dir, "decoder_norm", precision,
    )

    for name, module, example in [
        ("pose_proj", model.head_pose.proj, torch.randn(B, C_DECODER)),
        ("camera_proj", model.head_camera.proj, torch.randn(B, C_DECODER)),
        ("kp_posemb_2d", model.keypoint_posemb_linear, torch.randn(B, 70, 2)),
        ("kp_feat_linear", model.keypoint_feat_linear, torch.randn(B, 70, C_BACKBONE)),
        ("kp_posemb_3d", model.keypoint3d_posemb_linear, torch.randn(B, 70, 3)),
        ("init_to_token", model.init_to_token_mhr, torch.randn(B, 525)),
        ("prev_to_token", model.prev_to_token_mhr, torch.randn(B, 522)),
    ]:
        sizes[name] = _export(ModuleWrapper(module), (example,), output_dir, name, precision)

    return sizes


def export_aux_ops(output_dir, precision: str) -> dict:
    """Stateless helper ops used inside the loop (empty ``.bin`` files are expected)."""
    print(f"\n  [Aux ops] {precision.upper()}")
    sizes = {}
    sizes["grid_sample"] = _export(
        GridSampleModel(), (torch.randn(1, 1280, 32, 32), torch.randn(1, 70, 1, 2)),
        output_dir, "grid_sample", precision,
    )
    sizes["camera_projection"] = _export(
        CameraProjection(),
        (
            torch.randn(1, 70, 3), torch.randn(1, 3), torch.tensor([[320.0, 240.0]]),
            torch.tensor([200.0]), torch.tensor([[640.0, 480.0]]), torch.tensor([800.0]),
        ),
        output_dir, "camera_projection", precision,
    )
    sizes["full_to_crop"] = _export(
        FullToCrop(), (torch.randn(1, 70, 2), torch.randn(2, 3)),
        output_dir, "full_to_crop", precision,
    )
    return sizes


# Body pose conversion indices, mirroring mhr_utils.compact_cont_to_model_params_body.
_BODY_3DOF_ROT_IDXS = np.array([
    (0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20),
    (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39),
    (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79),
    (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106),
    (114, 98, 109), (115, 99, 103), (130, 131, 132),
], dtype=np.int64)
_BODY_1DOF_ROT_IDXS = np.array([
    1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50,
    51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80,
    81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111,
    116, 117, 118, 119, 120, 121, 122, 123,
], dtype=np.int64)
_BODY_1DOF_TRANS_IDXS = np.array([124, 125, 126, 127, 128, 129], dtype=np.int64)
_MHR_PARAM_HAND_IDXS = list(range(62, 116))
#: Degrees of freedom per hand joint, in model-parameter order.
_HAND_DOFS_IN_ORDER = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1]


def export_buffers(model, output_dir) -> None:
    """Write the static ``.npz`` tensors the OpenVINO pipeline needs.

    ``iterative_buffers.npz`` holds the decoder initialization (init pose/camera,
    prompt embedding, image positional encoding, keypoint embeddings).
    ``pose_head_buffers.npz`` holds everything needed to turn the raw 519-dim
    decoder output into MHR parameters, plus the mesh faces.
    """
    buffers_dir = Path(output_dir) / "buffers"
    buffers_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        dummy_kps = torch.zeros(1, 1, 3)
        dummy_kps[:, :, -1] = -2
        prompt_embed_raw, _ = model.prompt_encoder(keypoints=dummy_kps)
        prompt_embed = model.prompt_to_token(prompt_embed_raw)
        image_pe = model.prompt_encoder.get_dense_pe((32, 32))
        np.savez(
            buffers_dir / "iterative_buffers.npz",
            init_pose=model.init_pose.weight.data.cpu().numpy(),
            init_camera=model.init_camera.weight.data.cpu().numpy(),
            prompt_embed=prompt_embed.squeeze(0).cpu().numpy(),
            image_pe=image_pe.flatten(2).permute(0, 2, 1).cpu().numpy(),
            keypoint_embedding=model.keypoint_embedding.weight.data.cpu().numpy(),
            keypoint3d_embedding=model.keypoint3d_embedding.weight.data.cpu().numpy(),
        )
    print("    - buffers/iterative_buffers.npz")

    hp = model.head_pose
    mhr_param_hand_mask = np.zeros(133, dtype=bool)
    mhr_param_hand_mask[_MHR_PARAM_HAND_IDXS] = True

    def hand_mask(per_joint: int, dofs) -> np.ndarray:
        return np.concatenate(
            [np.ones(per_joint * k, dtype=bool) * (k in dofs) for k in _HAND_DOFS_IN_ORDER]
        )

    np.savez(
        buffers_dir / "pose_head_buffers.npz",
        scale_mean=hp.scale_mean.detach().cpu().numpy(),
        scale_comps=hp.scale_comps.detach().cpu().numpy(),
        hand_pose_mean=hp.hand_pose_mean.detach().cpu().numpy(),
        hand_pose_comps=hp.hand_pose_comps.detach().cpu().numpy(),
        hand_joint_idxs_left=hp.hand_joint_idxs_left.detach().cpu().numpy(),
        hand_joint_idxs_right=hp.hand_joint_idxs_right.detach().cpu().numpy(),
        keypoint_mapping=hp.keypoint_mapping.detach().cpu().numpy(),
        faces=hp.faces.detach().cpu().numpy(),
        mhr_param_hand_mask=mhr_param_hand_mask,
        body_3dof_rot_idxs=_BODY_3DOF_ROT_IDXS,
        body_1dof_rot_idxs=_BODY_1DOF_ROT_IDXS,
        body_1dof_trans_idxs=_BODY_1DOF_TRANS_IDXS,
        hand_mask_cont_3dof=hand_mask(2, {3}),
        hand_mask_cont_1dof=hand_mask(2, {1, 2}),
        hand_mask_model_3dof=hand_mask(1, {3}),
        hand_mask_model_1dof=hand_mask(1, {1, 2}),
    )
    print("    - buffers/pose_head_buffers.npz")


def export_openvino_ir(
    precision: str = "fp16",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    checkpoint: str = DEFAULT_CHECKPOINT,
    mhr_path: str = DEFAULT_MHR_PATH,
    model=None,
) -> None:
    """Export the whole pipeline to ``<output_dir>/<precision>/``.

    Pass an already-loaded ``model`` to export several precisions in one go.
    """
    prec_dir = Path(output_dir) / precision
    print(f"\n=== Exporting OpenVINO IR ({precision.upper()}) -> {prec_dir} ===")
    started = time.perf_counter()

    if model is None:
        model = load_reference_model(checkpoint, mhr_path)

    export_backbone(model, prec_dir / "backbone", precision)
    export_iterative_components(model, prec_dir / "iterative", precision)
    export_aux_ops(prec_dir / "iterative", precision)
    print("\n  [Buffers]")
    export_buffers(model, prec_dir / "iterative")
    export_mhr(mhr_path, prec_dir / "mhr", precision)
    export_mask_encoder(model, prec_dir / "mask_encoder", precision)

    print(f"\n=== {precision.upper()} export finished in {time.perf_counter() - started:.1f}s ===")


def main():
    parser = argparse.ArgumentParser(description="Export SAM 3D Body to OpenVINO IR.")
    parser.add_argument("--precision", nargs="+", default=["fp16"], choices=list(PRECISIONS))
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mhr_path", default=DEFAULT_MHR_PATH)
    args = parser.parse_args()

    model = load_reference_model(args.checkpoint, args.mhr_path)
    for precision in args.precision:
        export_openvino_ir(precision, args.output_dir, args.checkpoint, args.mhr_path, model=model)


if __name__ == "__main__":
    main()
