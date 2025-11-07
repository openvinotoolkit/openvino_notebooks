"""Post-training INT8 quantization for an OpenVINO ACT model.

This script performs INT8 post‑training quantization (static) using NNCF's

Requirements:
    pip install --upgrade openvino-dev nncf

Example:
    python quantize_int8_helper.py \
        --model_xml /path/to/model.xml \
        --stats_path /path/to/stats.json \
        --dataset_root /path/to/dataset/root \
        --output_dir /path/to/out_int8 --num_calib_samples 300

After success you will have model_int8.xml / model_int8.bin.
You can evaluate using your existing eval script by pointing to the INT8 model.
"""
from __future__ import annotations
import argparse
import os
import json
import numpy as np
from typing import List, Dict
import importlib.metadata as md
import warnings

from openvino.runtime import Core, serialize
import openvino as ov_pkg  # keep reference to top-level package for compatibility shims

# --- Early compatibility probe for openvino.op (required by newer NNCF OpenVINO backend) ---
def _check_openvino_op_module():
    """Detect presence of openvino.op module and provide actionable guidance if missing.

    Newer NNCF releases expect `import openvino.op as op`. This is available only in newer
    OpenVINO Python packages (2025.x or certain late 2024 builds). If absent we should abort
    before deep inside NNCF with a clear remediation path.
    """
    ov_version = getattr(ov_pkg, '__version__', 'unknown')
    try:
        import importlib
        importlib.import_module('openvino.op')  # noqa: F401
        return True, ov_version
    except Exception:
        return False, ov_version

_has_op, _ov_ver = _check_openvino_op_module()
if not _has_op:
    print('[compat] Missing module `openvino.op` (OpenVINO version:', _ov_ver, ')')
    print('[compat] Your installed NNCF likely requires a newer OpenVINO Python API exposing `openvino.op`.')
    print('[action] Choose ONE option and re-run this script:')
    print('  Option A (Recommended): Upgrade OpenVINO stack:')
    print("    pip install -U 'openvino>=2025.0.0'")
    print('  Option B: Downgrade NNCF to a version compatible with current OpenVINO (e.g. 2.16.0):')
    print("    pip install 'nncf<2.18'  # example: pip install nncf==2.16.0")
    print('[hint] After adjusting packages restart the kernel / environment, then run INT8 cell again.')
    # Abort early to avoid ModuleNotFoundError deeper inside NNCF
    raise SystemExit('Aborting INT8 quantization due to missing openvino.op module.')

# We will import nncf after applying an OpenVINO compatibility shim.
def _nncf_versions_report():
    try:
        nncf_ver = md.version('nncf')
    except md.PackageNotFoundError:
        nncf_ver = 'not-installed'
    ov_ver = getattr(ov_pkg, '__version__', 'unknown')
    return ov_ver, nncf_ver


def _apply_openvino_node_shim():
    """Provide openvino.Node alias if missing (newer NNCF expects it).

    """
    try:
        import openvino.runtime as ovrt
        if not hasattr(ov_pkg, 'Node') and hasattr(ovrt, 'Node'):
            ov_pkg.Node = ovrt.Node  # type: ignore[attr-defined]
            print("[compat] Injected openvino.Node alias -> openvino.runtime.Node")
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Failed to apply openvino.Node shim: {exc}")


def _ensure_version_alignment():
    ov_ver, nncf_ver = _nncf_versions_report()
    print(f"[info] OpenVINO version: {ov_ver} | NNCF version: {nncf_ver}")
    # Basic heuristics: if nncf >=2.18 but OpenVINO still 2024.*, warn user.
    try:
        from packaging.version import Version
        if nncf_ver not in ('not-installed', 'unknown'):
            if Version(nncf_ver) >= Version('2.18') and ov_ver.startswith('2024.'):
                print("[warn] Detected nncf >=2.18 with OpenVINO 2024.*. Consider either:\n"
                      "       Upgrade OpenVINO: pip install -U 'openvino-dev>=2025.3.0' 'openvino>=2025.3.0'\n"
                      "       OR downgrade NNCF: pip install 'nncf<2.18' (e.g. nncf==2.16.0)")
    except Exception:
        pass


def _import_nncf():
    try:
        from nncf import quantize, Dataset
        return quantize, Dataset
    except ImportError as e:
        raise SystemExit("nncf not installed. Install with: pip install nncf") from e

# LeRobot utilities (assuming project layout already on PYTHONPATH when run from repo root)
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.normalize import Normalize
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Failed to import LeRobot packages: {e}")


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def detect_camera_names(stats: dict) -> List[str]:
    return sorted({k.split('.')[-1] for k in stats.keys() if k.startswith('observation.images.')})


def build_normalizer(norm_stats: dict, camera_names: List[str], state_dim: int) -> Normalize:
    features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))}
    for cam in camera_names:
        features[f"observation.images.{cam}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
    }
    stats = {
        "observation.state": {
            "mean": np.asarray(norm_stats["observation.state"]["mean"], dtype=np.float32),
            "std": np.asarray(norm_stats["observation.state"]["std"], dtype=np.float32),
        }
    }
    for cam in camera_names:
        stats[f"observation.images.{cam}"] = {
            "mean": np.asarray(norm_stats[f"observation.images.{cam}"]["mean"], dtype=np.float32).reshape(3, 1, 1),
            "std": np.asarray(norm_stats[f"observation.images.{cam}"]["std"], dtype=np.float32).reshape(3, 1, 1),
        }
    return Normalize(features, norm_map, stats)


def infer_state_dim(stats: dict) -> int:
    return len(stats["observation.state"]["mean"])


from typing import Tuple


def derive_action_and_chunk(model, stats: dict) -> Tuple[int, int, bool]:
    """Infer (action_dim, chunk_size, has_action_inputs) with dynamic-shape safety.
    """
    action_dim = None
    chunk_size = None
    has_action_inputs = False

    # Pass 1: direct match on name == 'action'
    for inp in model.inputs:
        name = inp.get_any_name()
        try:
            pshape = inp.get_partial_shape()
        except Exception:
            continue
        if name == 'action' and pshape.is_static and len(pshape) == 3:
            shape = pshape.to_shape()
            chunk_size = int(shape[1])
            action_dim = int(shape[2])
            has_action_inputs = True
            break

    # Pass 2: generic 3D search (batch, chunk, action_dim)
    if action_dim is None:
        for inp in model.inputs:
            try:
                pshape = inp.get_partial_shape()
            except Exception:
                continue
            if not pshape.is_static or len(pshape) != 3:
                continue
            shape = pshape.to_shape()
            if shape[0] == 1:  # likely (1, chunk, action_dim)
                chunk_size = int(shape[1])
                action_dim = int(shape[2])
                # Determine if action_is_pad is present
                for i2 in model.inputs:
                    if i2.get_any_name() == 'action_is_pad':
                        has_action_inputs = True
                        break
                break

    # Fallback: stats.json
    if action_dim is None:
        if 'action' in stats and 'mean' in stats['action']:
            action_dim = len(stats['action']['mean'])
            # Prefer explicit chunk_size in stats if provided else 1
            chunk_size = int(stats.get('chunk_size', 1) or 1)
            has_action_inputs = False
            print(f"[info] No action inputs found in IR. Using stats fallback action_dim={action_dim}, chunk_size={chunk_size}.")
        else:
            raise ValueError(
                "Cannot infer action_dim: stats.json lacks action.mean and model inputs provide no static 3D tensor."
            )
    return action_dim, chunk_size, has_action_inputs


def build_sample(model_inputs, sample_step, normalizer: Normalize, camera_names: List[str], action_dim: int, chunk_size: int, has_action_inputs: bool) -> Dict[str, np.ndarray]:
    # Prepare observation dict for normalization
    obs = {}
    if "observation.state" in sample_step:
        state = sample_step["observation.state"].cpu().numpy().astype(np.float32)
        obs["observation.state"] = state
    for cam in camera_names:
        key = f"observation.images.{cam}"
        if key in sample_step:
            img = sample_step[key].cpu().numpy().astype(np.float32)
            if img.ndim == 3 and img.shape[0] == 3:
                pass
            elif img.ndim == 3 and img.shape[2] == 3:  # HWC -> CHW
                img = np.transpose(img, (2, 0, 1))
            obs[key] = img

    import torch
    tensor_input = {k: torch.from_numpy(v) for k, v in obs.items()}
    normed = normalizer(tensor_input)

    feed = {}
    # Map normalization results back to model input names
    for inp in model_inputs:
        name = inp.get_any_name()
        if name == 'observation_state' and 'observation.state' in normed:
            feed[name] = normed['observation.state'].unsqueeze(0).numpy()
        elif name.startswith('observation_images_'):
            idx = int(name.split('_')[-1])
            if idx < len(camera_names):
                cam_key = f"observation.images.{camera_names[idx]}"
                if cam_key in normed:
                    feed[name] = normed[cam_key].unsqueeze(0).numpy()
        elif has_action_inputs and name == 'action_is_pad':
            feed[name] = np.zeros((1, chunk_size), dtype=bool)
        elif has_action_inputs and name == 'action':
            feed[name] = np.zeros((1, chunk_size, action_dim), dtype=np.float32)
        elif name == 'observation_environment_state':
            # Dynamic-shape safe extraction of env dim
            env_dim = 1
            try:
                pshape = inp.get_partial_shape()
                if pshape.is_static:
                    shape = pshape.to_shape()
                    if len(shape) > 1:
                        env_dim = int(shape[1])
            except Exception:
                pass
            feed[name] = np.zeros((1, env_dim), dtype=np.float32)
    return feed


def collect_calibration_samples(core_model, dataset: LeRobotDataset, normalizer: Normalize, camera_names: List[str], action_dim: int, chunk_size: int, has_action_inputs: bool, num: int):
    inputs = core_model.inputs
    samples = []
    from_idx = dataset.episode_data_index['from'][0].item()
    to_idx = dataset.episode_data_index['to'][0].item()
    end = min(to_idx, from_idx + num)
    for idx in range(from_idx, end):
        step = dataset[idx]
        sample = build_sample(inputs, step, normalizer, camera_names, action_dim, chunk_size, has_action_inputs)
        samples.append(sample)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_xml', required=True, help='Path to FP32 OpenVINO model XML')
    ap.add_argument('--stats_path', required=True, help='Path to stats.json used for normalization')
    ap.add_argument('--dataset_root', required=True, help='Root of LeRobot dataset (local)')
    ap.add_argument('--output_dir', required=True, help='Directory to save INT8 model')
    ap.add_argument('--num_calib_samples', type=int, default=300, help='Number of calibration samples')
    ap.add_argument('--preset', choices=['performance', 'accuracy'], default='performance', help='Quantization preset')
    ap.add_argument('--action_dim', type=int, default=None, help='Override action dimension if inference fails')
    ap.add_argument('--chunk_size', type=int, default=None, help='Override chunk size if inference fails')
    ap.add_argument('--subset_size', type=int, default=None, help='Override subset size (defaults to num_calib_samples)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    core = Core()
    model = core.read_model(args.model_xml)

    # Load stats & dataset
    stats = load_json(args.stats_path)
    camera_names = detect_camera_names(stats)
    state_dim = infer_state_dim(stats)
    normalizer = build_normalizer(stats, camera_names, state_dim)
    dataset = LeRobotDataset(repo_id=None, root=args.dataset_root)

    action_dim, chunk_size, has_action_inputs = derive_action_and_chunk(model, stats)
    # CLI overrides
    if args.action_dim is not None:
        action_dim = args.action_dim
    if args.chunk_size is not None:
        chunk_size = args.chunk_size
    print(f"[info] has_action_inputs={has_action_inputs} action_dim={action_dim} chunk_size={chunk_size}")

    print(f"[info] Cameras={camera_names} state_dim={state_dim} action_dim={action_dim} chunk_size={chunk_size}")
    print(f"[info] Collecting {args.num_calib_samples} calibration samples ...")
    samples = collect_calibration_samples(model, dataset, normalizer, camera_names, action_dim, chunk_size, has_action_inputs, args.num_calib_samples)

    # Version alignment & compatibility shim before touching nncf internals
    _apply_openvino_node_shim()
    _ensure_version_alignment()
    quantize, Dataset = _import_nncf()

    # Wrap samples for NNCF Dataset (expects iterator over input dicts)
    nncf_dataset = Dataset(samples)
    subset = args.subset_size or len(samples)
    print(f"[info] Quantizing (preset={args.preset}, subset_size={subset}) ...")
    try:
        # Map string preset to QuantizationPreset enum for NNCF versions that expect enum (avoids AttributeError)
        preset_arg = args.preset
        try:  # lightweight, safe attempt
            from nncf.quantization import QuantizationPreset as _QPreset  # type: ignore
            if isinstance(preset_arg, str):
                preset_arg = _QPreset.PERFORMANCE if args.preset == 'performance' else _QPreset.ACCURACY
        except Exception:
            preset_arg = args.preset  # fall back to raw string
        quantized_model = quantize(model, nncf_dataset, preset=preset_arg, subset_size=subset)
    except AttributeError as attr_err:
        if 'openvino' in str(attr_err) and 'Node' in str(attr_err):
            print("[error] NNCF encountered missing openvino.Node despite shim. This indicates a deeper version mismatch.")
            print("[hint] Fix options: \n"
                  "  1) Upgrade OpenVINO stack: pip install -U 'openvino>=2025.3.0'\n"
                  "  2) Downgrade NNCF: pip install 'nncf<2.18' (e.g. nncf==2.16.0)\n"
                  "Re-run this script after adjusting versions.")
        raise

    out_xml = os.path.join(args.output_dir, 'model_int8.xml')
    out_bin = os.path.join(args.output_dir, 'model_int8.bin')
    serialize(quantized_model, out_xml, out_bin)
    print(f"[done] INT8 model saved to: {out_xml} / {out_bin}")
    print("Evaluate it with your evaluation script pointing to model_int8.xml")


if __name__ == '__main__':
    main()
