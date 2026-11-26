"""System 2 trajectory-latent export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov
from safetensors import safe_open

from .checkpoint import read_weight_map

LATENT_QUERY_KEY = "model.latent_queries"
TRAJECTORY_TOKEN_ID = 151667
TRAJECTORY_QUERY_COUNT = 4


def load_latent_queries(model_dir: Path) -> np.ndarray:
    weight_map = read_weight_map(model_dir)
    shard = weight_map.get(LATENT_QUERY_KEY)
    if shard is None:
        raise KeyError(f"checkpoint contains no tensor {LATENT_QUERY_KEY!r}")
    with safe_open(model_dir / shard, framework="pt", device="cpu") as tensors:
        latent_queries = tensors.get_tensor(LATENT_QUERY_KEY)
    expected_shape = (1, TRAJECTORY_QUERY_COUNT, 3584)
    if latent_queries.shape != expected_shape:
        raise ValueError(f"unexpected latent query shape: {latent_queries.shape}, expected {expected_shape}")
    return latent_queries.float().numpy()


def add_trajectory_latent_output(model: ov.Model) -> None:
    final_norm_output = [op for op in model.get_ordered_ops() if op.get_friendly_name().endswith("model.language_model.norm/aten::mul/Multiply_1")]
    if len(final_norm_output) != 1:
        raise RuntimeError("could not uniquely locate the language model's final RMSNorm output: " f"found {len(final_norm_output)} candidates")
    trajectory_latents = ov.opset13.slice(
        final_norm_output[0].output(0),
        np.array([-TRAJECTORY_QUERY_COUNT], dtype=np.int64),
        np.array([np.iinfo(np.int64).max], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([1], dtype=np.int64),
    )
    trajectory_latents.set_friendly_name("trajectory_latents_slice")
    trajectory_latents.output(0).get_tensor().set_names({"trajectory_latents"})
    existing = [output for output in model.outputs if "trajectory_latents" in output.get_names()]
    if len(existing) > 1:
        raise RuntimeError("multiple trajectory_latents outputs")
    if existing:
        existing[0].get_node().input(0).replace_source_output(trajectory_latents.output(0))
    else:
        model.add_outputs(trajectory_latents.output(0))
    model.validate_nodes_and_infer_types()
