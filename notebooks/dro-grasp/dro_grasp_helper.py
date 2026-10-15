import time
from collections.abc import Callable, Sequence

import numpy as np
import torch


class OpenVINODROGrasp:
    """Compile and run a DRO-Grasp OpenVINO model."""

    def __init__(self, core, model, device: str = "CPU", config: dict | None = None):
        self.compiled_model = core.compile_model(model, device, config or {"PERFORMANCE_HINT": "LATENCY"})

    def __call__(self, robot_pc: np.ndarray, object_pc: np.ndarray, latent: np.ndarray) -> np.ndarray:
        if robot_pc.shape != object_pc.shape or robot_pc.ndim != 3 or robot_pc.shape[-1] != 3:
            raise ValueError("robot_pc and object_pc must have matching [batch, points, 3] shapes")
        if latent.ndim != 2 or latent.shape[0] != robot_pc.shape[0]:
            raise ValueError("latent must have shape [batch, latent_dim]")
        return self.compiled_model({"robot_pc": robot_pc, "object_pc": object_pc, "latent": latent})["dro"]


class DROGraspValidationWrapper(torch.nn.Module):
    """Expose deterministic DRO-Grasp validation as a tensor-only graph."""

    def __init__(self, network: torch.nn.Module):
        super().__init__()
        self.network = network

    def forward(self, robot_pc: torch.Tensor, object_pc: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.network(robot_pc=robot_pc, object_pc=object_pc, target_pc=None, latent=latent)["dro"]


def prepare_openvino_model(ov_model):
    """Assign stable names to the converted graph inputs and output."""
    for model_input, name in zip(ov_model.inputs, ("robot_pc", "object_pc", "latent")):
        model_input.get_tensor().set_names({name})
    ov_model.output(0).get_tensor().set_names({"dro"})
    return ov_model


def comparison_metrics(reference: np.ndarray, result: np.ndarray) -> dict[str, float]:
    absolute_error = np.abs(reference - result)
    reference_flat = reference.reshape(-1).astype(np.float64)
    result_flat = result.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(reference_flat) * np.linalg.norm(result_flat)
    cosine_similarity = float(np.dot(reference_flat, result_flat) / denominator) if denominator else 1.0
    return {
        "max_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
        "cosine_similarity": cosine_similarity,
    }


def benchmark(
    infer: Callable[[], object],
    iterations: int = 20,
    warmup_iterations: int = 3,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, float]:
    for _ in range(warmup_iterations):
        infer()
    if synchronize:
        synchronize()

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        infer()
        if synchronize:
            synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    latency_array = np.asarray(latencies)
    mean_latency = float(latency_array.mean())
    return {
        "mean_latency_ms": mean_latency,
        "median_latency_ms": float(np.median(latency_array)),
        "p95_latency_ms": float(np.percentile(latency_array, 95)),
        "throughput_fps": 1000.0 / mean_latency,
    }


def available_inference_devices(core, requested_devices: Sequence[str] = ("CPU", "GPU")) -> list[str]:
    available_types = {device.split(".")[0] for device in core.available_devices}
    return [device for device in requested_devices if device in available_types]
