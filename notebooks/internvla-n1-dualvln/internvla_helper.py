"""Inputs and a self-contained PyTorch reference for the InternVLA-N1 DualVLN notebook.

The upstream project produces reference tensors with ``run_baseline.py``, which imports the
InternNav source tree (and its Depth Anything / LongCLIP dependencies) and pins an older
``transformers``. None of that is needed at inference time, so this module rebuilds the
*inference* path straight from the checkpoint with the same ``internvla`` modules that
``export.py`` converts. Comparing the two therefore measures exactly the cost of the
conversion.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from export import prepare_trajectory_latent_inputs
from internvla.checkpoint import load_qwen_model
from internvla.inference import split_prompt
from internvla.system1 import load_system1_context_encoder, load_system1_denoiser

# The prompt InternNav's agent sends to System 2 on every front-camera observation.
BASE_PROMPT = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully "
    "completed the task."
)

ACTION_NAMES = {0: "STOP", 1: "move forward 25 cm", 2: "turn left 15°", 3: "turn right 15°", 5: "look down 30°"}

S2_IMAGE_SIZE = 384
S1_IMAGE_SIZE = 224
S1_SAMPLES = 32
S1_HORIZON = 32
S1_STEPS = 10


def make_fixed_observations(num_history: int = 2, seed: int = 0) -> list[Image.Image]:
    """Deterministic stand-in camera frames: ``num_history`` past views plus the current one."""
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 256, (S2_IMAGE_SIZE, S2_IMAGE_SIZE, 3), dtype=np.uint8)) for _ in range(num_history + 1)]


def build_messages(instruction: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    """Single-turn System 2 conversation: instruction, history frames, then the current frame."""
    prompt = BASE_PROMPT.replace("<instruction>.", instruction)
    if len(images) > 1:
        prompt += f" These are your historical observations: {'<image>\n' * (len(images) - 1)}."
    prompt += " you can see <image>."
    content: list[dict[str, Any]] = []
    image_index = 0
    for part in split_prompt(prompt):
        if part == "<image>":
            content.append({"type": "image", "image": images[image_index]})
            image_index += 1
        else:
            content.append({"type": "text", "text": part})
    return [{"role": "user", "content": content}]


def load_processor(model_dir: Path):
    processor = AutoProcessor.from_pretrained(model_dir, use_fast=False)
    processor.tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    processor.tokenizer.padding_side = "left"
    return processor


def preprocess(processor, messages: list[dict[str, Any]], images: list[Image.Image]) -> dict[str, np.ndarray]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processed = processor(text=[text], images=images, return_tensors="np")
    return {name: np.asarray(processed[name]) for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}


def system1_images(goal_image: Image.Image, current_image: Image.Image) -> np.ndarray:
    """System 1 sees the frame the pixel goal was chosen in and the latest frame, as [1, 2, H, W, 3] in [0, 1]."""
    frames = [np.asarray(image.resize((S1_IMAGE_SIZE, S1_IMAGE_SIZE)), dtype=np.float32) / 255.0 for image in (goal_image, current_image)]
    return np.stack(frames)[None]


def initial_noise(seed: int = 0) -> np.ndarray:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((S1_SAMPLES, S1_HORIZON, 3), dtype=torch.float32, generator=generator).numpy()


def flow_match_sigmas(num_steps: int = S1_STEPS) -> np.ndarray:
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps, dtype=np.float32)
    return np.concatenate((sigmas, np.zeros(1, dtype=np.float32)))


def denoiser_step_inputs(trajectories: np.ndarray, sigma: float, context: np.ndarray) -> dict[str, np.ndarray]:
    """Classifier-free-guidance batch: first half with an all-zero context, second half conditioned."""
    model_input = np.concatenate((trajectories, trajectories), axis=0)
    denoiser_context = np.repeat(np.concatenate((np.zeros_like(context), context), axis=0), trajectories.shape[0], axis=0)
    return {
        "trajectories": model_input,
        "timestep": np.full(model_input.shape[0], round(float(sigma) * 1000), dtype=np.int64),
        "context": denoiser_context,
    }


def trajectory_paths(trajectories: np.ndarray) -> np.ndarray:
    """Integrate the predicted per-step deltas into [samples, horizon + 1, 2] paths in metres (x forward, y left)."""
    deltas = trajectories[:, :, :2] / 4.0
    paths = np.zeros((len(deltas), deltas.shape[1] + 1, 2), dtype=np.float64)
    paths[:, 1:] = np.cumsum(deltas, axis=1)
    return paths


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def rel_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    reference = np.asarray(reference, np.float64)
    return float(np.linalg.norm(np.asarray(actual, np.float64) - reference) / max(np.linalg.norm(reference), 1e-12))


def generate_reference(
    out_dir: Path,
    model_dir: Path,
    instruction: str,
    images: list[Image.Image],
    seed: int = 0,
    max_new_tokens: int = 128,
) -> dict[str, np.ndarray]:
    """Run the full DualVLN pipeline in PyTorch FP32 on CPU and save every boundary tensor."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = load_processor(model_dir)
    inputs = preprocess(processor, build_messages(instruction, images), images)
    torch_inputs = {name: torch.from_numpy(value) for name, value in inputs.items()}

    print("  running Qwen2.5-VL System 2 in PyTorch (fp32, CPU) ...")
    qwen = load_qwen_model(model_dir, dtype=torch.float32)
    with torch.no_grad():
        generated_ids = qwen.generate(**torch_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Trajectory latents are always produced so System 1 can be validated whichever branch is chosen.
    _, trajectory_latents, _ = prepare_trajectory_latent_inputs(qwen, generated_ids, torch_inputs["pixel_values"], torch_inputs["image_grid_thw"], model_dir)
    del qwen
    gc.collect()

    print("  running System 1 in PyTorch ...")
    images_dp = system1_images(images[-1], images[-1])
    noise = initial_noise(seed)
    encoder = load_system1_context_encoder(model_dir)
    denoiser = load_system1_denoiser(model_dir)
    tensors: dict[str, np.ndarray] = {}
    with torch.no_grad():
        context = encoder(torch.from_numpy(images_dp.transpose(0, 1, 4, 2, 3).copy()), torch.from_numpy(trajectory_latents)).numpy()
        trajectories = noise.copy()
        sigmas = flow_match_sigmas()
        for i, sigma in enumerate(sigmas[:-1]):
            step_inputs = denoiser_step_inputs(trajectories, sigma, context)
            prediction = denoiser(*(torch.from_numpy(step_inputs[name]) for name in ("trajectories", "timestep", "context"))).numpy()
            tensors[f"s1_trajectories_step{i}"] = trajectories.copy()
            tensors[f"s1_noise_prediction_step{i}"] = prediction
            unconditional, conditional = np.split(prediction, 2)
            # Guidance scale 1.0, written out as in System1Runner so both sides round identically.
            trajectories = trajectories + (sigmas[i + 1] - sigma) * (unconditional + 1.0 * (conditional - unconditional))
    del encoder, denoiser
    gc.collect()

    tensors.update(
        {
            **inputs,
            "generated_ids": generated_ids.numpy(),
            "trajectory_latents": trajectory_latents,
            "images_dp": images_dp,
            "initial_noise": noise,
            "s1_context": context,
            "trajectories": trajectories,
        }
    )
    for name, value in tensors.items():
        np.save(out_dir / f"{name}.npy", np.asarray(value))
    print(f"  wrote {len(tensors)} tensors to {out_dir}")
    return tensors
