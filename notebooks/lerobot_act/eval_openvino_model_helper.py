"""Model action comparison."""

import logging
import time
import json
import sys
import os
from dataclasses import asdict
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from openvino.runtime import Core

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.utils import get_safe_torch_device, init_logging
from unitree_lerobot.eval_robot.utils.utils import (
    extract_observation,
    predict_action,
    EvalRealConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_STATS_PATH = "/G1_BlockStacking_Dataset/meta/stats.json"
DEFAULT_OPENVINO_MODEL_PATH = "/path_to_model.xml"

DEFAULT_CHUNK_STRATEGY = "first"  # options: first, mean
DEFAULT_TEMPORAL_ENSEMBLE_COEFF = 0.01
DEFAULT_CHUNK_SIZE = 100

# Set OPENVINO_MODEL_PATH and STATS_PATH before invoking this script.
# Falls back to defaults if env vars not present.
OPENVINO_MODEL_ENV = os.getenv("OPENVINO_MODEL_PATH")
STATS_PATH_ENV = os.getenv("STATS_PATH")
OPENVINO_DEVICE_ENV = os.getenv("OPENVINO_DEVICE") or os.getenv("OV_DEVICE") or "CPU"
ALLOWED_OPENVINO_DEVICES = {"CPU", "GPU", "NPU", "AUTO"}

#########################
# OpenVINO Helper Logic #
#########################
def load_norm_stats(stats_path: str):
    with open(stats_path, "r") as f:
        return json.load(f)


def detect_camera_keys(norm_stats: dict):
    return sorted({k.split(".")[-1] for k in norm_stats.keys() if k.startswith("observation.images.")})


def build_normalizer(norm_stats: dict, camera_names, state_dim: int):
    features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))}
    for cam in camera_names:
        features[f"observation.images.{cam}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    stats = {
        "observation.state": {
            "mean": torch.tensor(norm_stats["observation.state"]["mean"], dtype=torch.float32),
            "std": torch.tensor(norm_stats["observation.state"]["std"], dtype=torch.float32),
        }
    }
    for cam in camera_names:
        stats[f"observation.images.{cam}"] = {
            "mean": torch.tensor(norm_stats[f"observation.images.{cam}"]["mean"], dtype=torch.float32).reshape(3, 1, 1),
            "std": torch.tensor(norm_stats[f"observation.images.{cam}"]["std"], dtype=torch.float32).reshape(3, 1, 1),
        }
    return Normalize(features, norm_map, stats)


def build_unnormalizer(norm_stats: dict):
    action_shape = (len(norm_stats["action"]["mean"]),)
    features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape)}
    norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
    stats = {
        "action": {
            "mean": torch.tensor(norm_stats["action"]["mean"], dtype=torch.float32),
            "std": torch.tensor(norm_stats["action"]["std"], dtype=torch.float32),
        }
    }
    return Unnormalize(features, norm_map, stats)

def predict_action_openvino(observation, compiled_model, input_names, normalizer, camera_names, state_dim):
    inputs = {}
    # Prepare input dict for normalization
    state = observation.get("observation.state", None)
    input_dict = {}
    if state is not None:
        if hasattr(state, 'cpu'):
            state = state.cpu().numpy().astype(np.float32)
        state = state.reshape(1, -1)
        input_dict["observation.state"] = torch.from_numpy(state)
    for cam in camera_names:
        img = observation.get(f"observation.images.{cam}", None)
        if img is not None:
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy().astype(np.float32)
            if img.ndim == 3 and img.shape[2] == 3:
                img = np.transpose(img, (2, 0, 1))
            if img.shape[-2:] != (480, 640):
                from cv2 import resize
                img = np.transpose(img, (1, 2, 0)) if img.shape[0] == 3 else img
                img = resize(img, (640, 480))
                img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32)
            input_dict[f"observation.images.{cam}"] = torch.from_numpy(img)
    # Normalize
    normed = normalizer(input_dict)
    if "observation.state" in normed:
        inputs["observation_state"] = normed["observation.state"].numpy()
    else:
        inputs["observation_state"] = np.zeros((1, state_dim), dtype=np.float32)
    for i, cam in enumerate(camera_names):
        key = f"observation.images.{cam}"
        if key in normed:
            img = normed[key]
            img = img[None, ...].numpy()
            inputs[f"observation_images_{i}"] = img
        else:
            inputs[f"observation_images_{i}"] = np.zeros((1, 3, 480, 640), dtype=np.float32)
    result = compiled_model(inputs)
    action = result[list(result.keys())[0]]
    return action.squeeze(0)

# ---------------- Prediction Wrapper ---------------- #
def predict_action_safetensor(observation, policy, device, use_amp, use_dataset=True):
    """Return numpy action predicted by the PyTorch (safetensor) policy."""
    act = predict_action(observation, policy, device, use_amp, use_dataset=use_dataset)
    return act.detach().cpu().numpy()

#############################
# (Optional) Temporal Smoothing
#############################
class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int):
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: np.ndarray) -> np.ndarray:
        actions = torch.from_numpy(actions)
        self.ensemble_weights = self.ensemble_weights.to(actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones((self.chunk_size, 1), dtype=torch.long, device=actions.device)
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat([
                self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])
            ])
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action.cpu().numpy()


# ---------------- Core Loop ---------------- #
def run_comparison_loop(cfg, dataset, policy, ov_ctx=None):
    """Iterate over dataset steps and collect model + ground truth actions.

    Returns (actions_dict, ground_truth_actions).
    """
    device = get_safe_torch_device(policy.config.device)
    use_amp = getattr(policy.config, "use_amp", False)

    from_idx = dataset.episode_data_index["from"][0].item()
    to_idx = dataset.episode_data_index["to"][0].item()

    safetensor_actions = []
    openvino_actions = [] if ov_ctx is not None else None
    if ov_ctx is not None:
        compiled_model = ov_ctx["compiled_model"]
        input_names = ov_ctx.get("input_names", [])
        normalizer = ov_ctx["normalizer"]
        unnormalizer = ov_ctx["unnormalizer"]
        camera_names = ov_ctx["camera_names"]
        state_dim = ov_ctx["state_dim"]
        chunk_size = ov_ctx.get("chunk_size")
        ensembler = ov_ctx.get("ensembler")
        chunk_strategy = ov_ctx.get("chunk_strategy", DEFAULT_CHUNK_STRATEGY)
    gt_actions = []

    for i in tqdm.tqdm(range(from_idx, to_idx)):
        loop_start = time.perf_counter()
        step = dataset[i]
        obs = extract_observation(step)
        safetensor_actions.append(predict_action_safetensor(obs, policy, device, use_amp, use_dataset=True))
        if ov_ctx is not None:
            # New signature call path
            ov_pred = predict_action_openvino(
                obs,
                compiled_model=compiled_model,
                input_names=input_names,
                normalizer=normalizer,
                camera_names=camera_names,
                state_dim=state_dim,
            )
            # Ensure numpy array
            if isinstance(ov_pred, torch.Tensor):
                norm_arr = ov_pred.detach().cpu().numpy()
            else:
                norm_arr = np.asarray(ov_pred)

            # If model outputs a temporal chunk, optionally ensemble or reduce before unnormalization
            if norm_arr.ndim == 2 and norm_arr.shape[0] > 1:  # (chunk, A)
                if ensembler is not None and chunk_size and norm_arr.shape[0] == chunk_size:
                    try:
                        norm_arr = ensembler.update(norm_arr[None, ...])  # returns (1, A) typically
                    except Exception:
                        pass  # fall back to strategy below
                if norm_arr.ndim == 2 and norm_arr.shape[0] > 1:  # still unresolved (chunk, A)
                    if chunk_strategy == "mean":
                        norm_arr = norm_arr.mean(axis=0)
                    else:
                        norm_arr = norm_arr[0]

            # Unnormalize (expects dict with action tensor)
            norm_tensor = torch.from_numpy(norm_arr.astype(np.float32))
            try:
                unnorm = unnormalizer({"action": norm_tensor})["action"].numpy()
            except Exception:
                unnorm = norm_tensor.numpy()

            # Squeeze potential leading singleton dimensions
            while unnorm.ndim > 1 and unnorm.shape[0] == 1:
                unnorm = unnorm.squeeze(0)
            if unnorm.ndim > 2:  # unexpected extra dims -> flatten last
                unnorm = unnorm.reshape(-1)[: norm_tensor.shape[-1]]
            openvino_actions.append(unnorm)
        gt = step["action"]
        if hasattr(gt, "cpu"):
            gt = gt.cpu().numpy()
        gt_actions.append(gt)
        if getattr(cfg, "frequency", None):
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start)))
    gt_np = np.asarray(gt_actions)
    actions_dict = {"safetensor": np.asarray(safetensor_actions)}
    if openvino_actions is not None:
        ov_arr = np.asarray(openvino_actions, dtype=np.float32)
        # Collapse shapes like (T,1,chunk,D) or (T,1,D)
        if ov_arr.ndim == 4 and ov_arr.shape[1] == 1:  # (T,1,chunk,D)
            ov_arr = ov_arr[:, 0, :, :]
        if ov_arr.ndim == 3:
            # (T,chunk,D) -> reduce chunk
            strategy = ov_ctx.get("chunk_strategy", DEFAULT_CHUNK_STRATEGY) if ov_ctx else DEFAULT_CHUNK_STRATEGY
            if ov_arr.shape[1] > 1:
                ov_arr = ov_arr.mean(axis=1) if strategy == "mean" else ov_arr[:, 0, :]
            else:  # (T,1,D)
                ov_arr = ov_arr[:, 0, :]
        actions_dict["openvino"] = ov_arr
    return actions_dict, gt_np


# ---------------- Plotting ---------------- #
def plot_comparison(actions_dict, ground_truth_actions, out_path="actions_comparison.png"):
    """Plot actions side-by-side: Left Arm joints in left column, Right Arm joints in right column.

    - Joint names + per-model μ/σ in each subplot.
    - If joint names don't include left/right, first half assumed left, second half right.
    - Handles unequal counts by leaving blank cells.
    """
    if not actions_dict:
        logger.warning("No actions to plot.")
        return
    sample = next(iter(actions_dict.values()))
    _, n_dims = sample.shape
    try:
        from unitree_lerobot.utils.constants import G1_INSPIRE_CONFIG
        joint_names = G1_INSPIRE_CONFIG.motors
        if len(joint_names) != n_dims:
            joint_names = [f"Joint {i+1}" for i in range(n_dims)]
    except Exception:
        joint_names = [f"Joint {i+1}" for i in range(n_dims)]

    preferred_order = [m for m in ["safetensor", "openvino"] if m in actions_dict]
    others = [m for m in actions_dict.keys() if m not in preferred_order]
    models = preferred_order + others

    colors = ["red", "green", "orange", "purple", "brown", "cyan"]
    styles = [":", "--", "-.", "-", (0, (3,1,1,1)), (0, (5,2))]

    stats = {}
    for m in models:
        err = actions_dict[m] - ground_truth_actions
        stats[m] = (np.mean(err, axis=0), np.std(err, axis=0))

    left_indices, right_indices = [], []
    for idx, name in enumerate(joint_names):
        lname = name.lower()
        if "left" in lname:
            left_indices.append(idx)
        elif "right" in lname:
            right_indices.append(idx)
    if not left_indices and not right_indices:
        half = n_dims // 2
        left_indices = list(range(half))
        right_indices = list(range(half, n_dims))
    if not left_indices:
        left_indices = [i for i in range(n_dims) if i not in right_indices]
    if not right_indices:
        right_indices = [i for i in range(n_dims) if i not in left_indices]

    n_rows = max(len(left_indices), len(right_indices))
    n_cols = 2 if right_indices else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14 if n_cols==2 else 7, 3.0 * n_rows), sharex=False)
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]

    fig.suptitle("Action Comparison (Left vs Right Arm)", fontsize=12)

    TITLE_FS = 10
    LABEL_FS = 12
    TICK_FS = 12

    def _plot_side(side_indices, col, side_name):
        for row in range(n_rows):
            ax = axes[row][col] if n_cols == 2 else axes[row][0]
            if row >= len(side_indices):
                ax.axis('off')
                continue
            j_idx = side_indices[row]
            ax.plot(ground_truth_actions[:, j_idx], label="Ground Truth", color="blue", linewidth=1.2)
            for k, m in enumerate(models):
                ax.plot(
                    actions_dict[m][:, j_idx],
                    label=f"{m} (μ={stats[m][0][j_idx]:.3f}, σ={stats[m][1][j_idx]:.3f})",
                    color=colors[k % len(colors)],
                    linestyle=styles[k % len(styles)],
                    linewidth=1.0,
                )
            summary_parts = [f"{m}:μ={stats[m][0][j_idx]:.2f} σ={stats[m][1][j_idx]:.2f}" for m in models[:2]]
            ax.set_title(f"{side_name} - {joint_names[j_idx]} ({' | '.join(summary_parts)})", fontsize=TITLE_FS)
            ax.set_ylabel("Val", fontsize=LABEL_FS)
            ax.grid(alpha=0.25, linestyle=":")
            ax.tick_params(axis='both', labelsize=TICK_FS)
            if row == 0:
                ax.legend(fontsize=7, ncol=2, loc="upper right")
            if row == n_rows - 1:
                ax.set_xlabel("Timestep", fontsize=LABEL_FS)
                ax.tick_params(axis='both', labelsize=TICK_FS)

    _plot_side(left_indices, 0, "Left Arm")
    if n_cols == 2:
        _plot_side(right_indices, 1, "Right Arm")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path)
    logger.info(
        f"Saved comparison figure to {out_path} (grid {n_rows}x{n_cols}; left joints={len(left_indices)}, right joints={len(right_indices)})"
    )


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    dataset = LeRobotDataset(
        repo_id=None if (cfg.repo_id is None or str(cfg.repo_id).lower() == "none") else cfg.repo_id,
        root=cfg.root,
    )
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

 
    # Optional OpenVINO setup
    import os
    ov_model_path = OPENVINO_MODEL_ENV or DEFAULT_OPENVINO_MODEL_PATH
    stats_path = STATS_PATH_ENV or DEFAULT_STATS_PATH
    logging.info(f"Using OpenVINO model path: {ov_model_path}; stats path: {stats_path} (env model={OPENVINO_MODEL_ENV is not None}, env stats={STATS_PATH_ENV is not None})")
    if not os.path.exists(ov_model_path):
        logging.warning(f"OpenVINO model path does not exist: {ov_model_path}. Skipping OpenVINO.")
    if not os.path.exists(stats_path):
        logging.warning(f"Stats path does not exist: {stats_path}. Skipping OpenVINO.")
    state_dim_arg = parser.parse_arg("state_dim")

    ov_ctx = None
    if os.path.exists(ov_model_path) and os.path.exists(stats_path):
        try:
            if state_dim_arg:
                state_dim = int(state_dim_arg)
            else:
                # Try infer from stats
                try:
                    stats_preview = load_norm_stats(stats_path)
                    state_dim = len(stats_preview["observation.state"]["mean"]) if "observation.state" in stats_preview else None
                except Exception:
                    state_dim = None
                if state_dim is None:
                    # Try dataset meta
                    try:
                        meta_state = dataset.meta["observation"]["state"]
                        if isinstance(meta_state, dict) and "shape" in meta_state:
                            shape = meta_state["shape"]
                            if isinstance(shape, (list, tuple)):
                                state_dim = shape[0]
                    except Exception:
                        pass
                if state_dim is None:
                    raise ValueError("Could not determine state_dim (provide --state_dim).")
            norm_stats = load_norm_stats(stats_path)
            camera_names = detect_camera_keys(norm_stats)
            normalizer = build_normalizer(norm_stats, camera_names, state_dim)
            unnormalizer = build_unnormalizer(norm_stats)
            core = Core()
            model = core.read_model(ov_model_path)
            ov_device = OPENVINO_DEVICE_ENV.upper()
            if ov_device not in ALLOWED_OPENVINO_DEVICES:
                logging.warning(
                    "Requested OPENVINO_DEVICE %s not in allowed %s; falling back to CPU.",
                    ov_device,
                    sorted(ALLOWED_OPENVINO_DEVICES),
                )
                ov_device = "CPU"
            precision_env = os.getenv("OPENVINO_PRECISION_HINT") or os.getenv("OV_PRECISION")
            if precision_env is None:
                lower_path = ov_model_path.lower()
                if "int8" in lower_path:
                    precision_env = "INT8"
                elif "fp16" in lower_path:
                    precision_env = "FP16"
                else:
                    precision_env = "FP32"
            precision_env = precision_env.upper()
            allowed_precisions = {"FP32", "FP16", "INT8"}
            if precision_env not in allowed_precisions:
                logging.warning("Invalid OPENVINO_PRECISION_HINT=%s; falling back to FP32.", precision_env)
                precision_env = "FP32"
            compile_config = {"INFERENCE_PRECISION_HINT": precision_env}
            logging.info(
                "Compiling OpenVINO model for device=%s with INFERENCE_PRECISION_HINT=%s", ov_device, precision_env
            )
            try:
                compiled_model = core.compile_model(model, ov_device, config=compile_config)
            except Exception as e:
                logging.warning(
                    "Precision-specific compile failed (%s). Retrying without config: %s", compile_config, e
                )
                compiled_model = core.compile_model(model, ov_device)
            try:
                input_names = [inp.get_any_name() for inp in model.inputs]
            except Exception:
                input_names = []
            ov_ctx = {
                "compiled_model": compiled_model,
                "camera_names": camera_names,
                "normalizer": normalizer,
                "unnormalizer": unnormalizer,
                "state_dim": state_dim,
                "input_names": input_names,
                    "chunk_size": DEFAULT_CHUNK_SIZE,
                    "ensembler": ACTTemporalEnsembler(DEFAULT_TEMPORAL_ENSEMBLE_COEFF, DEFAULT_CHUNK_SIZE),
                    "chunk_strategy": DEFAULT_CHUNK_STRATEGY,
            }
            logging.info(
                "OpenVINO model loaded: %s (device=%s, precision=%s, cameras=%s, state_dim=%s, temporal_ensemble=on, chunk_size=%d, coeff=%.4f)",
                ov_model_path,
                ov_device,
                precision_env,
                camera_names,
                state_dim,
                DEFAULT_CHUNK_SIZE,
                DEFAULT_TEMPORAL_ENSEMBLE_COEFF,
            )
        except Exception as e:
            logging.warning(f"Failed to initialize OpenVINO path '{ov_model_path}': {e}")

    actions_dict, gt_actions = run_comparison_loop(cfg, dataset, policy, ov_ctx=ov_ctx)
    if "openvino" not in actions_dict:
        logging.warning("OpenVINO actions not collected; only plotting sensorsafe model.")
    else:
        logging.info("Models plotted: %s", list(actions_dict.keys()))
    plot_comparison(actions_dict, gt_actions)
    logging.info("Evaluation complete")


if __name__ == "__main__":
    init_logging()
    eval_main()