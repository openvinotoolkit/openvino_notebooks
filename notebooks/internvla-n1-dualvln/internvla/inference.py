"""OpenVINO inference orchestration for InternVLA-N1 DualVLN."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
import torch
from optimum.intel.openvino import OVModelForVisualCausalLM
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from .system2 import TRAJECTORY_QUERY_COUNT, TRAJECTORY_TOKEN_ID

ACTION_MAP = {"STOP": 0, "↑": 1, "←": 2, "→": 3, "↓": 5}


def compare_arrays(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference64 = reference.astype(np.float64)
    actual64 = actual.astype(np.float64)
    delta = actual64 - reference64
    reference_norm = max(float(np.linalg.norm(reference64.ravel())), 1e-12)
    cosine_norm = max(
        float(np.linalg.norm(reference64.ravel()) * np.linalg.norm(actual64.ravel())),
        1e-12,
    )
    return {
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "relative_l2_error": float(np.linalg.norm(delta.ravel()) / reference_norm),
        "cosine_similarity": float(np.dot(reference64.ravel(), actual64.ravel()) / cosine_norm),
    }


def parse_generated_text(text: str) -> dict[str, Any]:
    digits = [int(value) for value in re.findall(r"\d+", text)]
    if digits:
        if len(digits) < 2:
            raise ValueError(f"pixel-goal output has fewer than two coordinates: {text!r}")
        return {"branch": "pixel_goal", "pixel_goal_yx": [digits[1], digits[0]]}
    pattern = "|".join(re.escape(token) for token in ACTION_MAP)
    return {
        "branch": "action",
        "actions": [ACTION_MAP[token] for token in re.findall(pattern, text)],
    }


def split_prompt(text: str) -> list[str]:
    parts = re.split(r"(<image>)", text)
    return [part.replace("\n", "").strip() for part in parts if part.strip()]


def qwen_image_rope_position_ids(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray,
    *,
    image_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> np.ndarray:
    position_ids = np.ones((3, input_ids.shape[0], input_ids.shape[1]), dtype=np.int64)
    image_index = 0
    for batch_index, sequence in enumerate(input_ids):
        vision_starts = np.flatnonzero(sequence == vision_start_token_id)
        image_count = int(np.sum(sequence[vision_starts + 1] == image_token_id))
        tokens = sequence.tolist()
        sections: list[np.ndarray] = []
        start = 0
        for _ in range(image_count):
            image_start = tokens.index(image_token_id, start)
            temporal, height, width = image_grid_thw[image_index]
            image_index += 1
            temporal = int(temporal)
            height = int(height) // spatial_merge_size
            width = int(width) // spatial_merge_size
            text_length = image_start - start
            section_start = int(sections[-1].max()) + 1 if sections else 0
            sections.append(np.broadcast_to(np.arange(text_length, dtype=np.int64)[None], (3, text_length)).copy() + section_start)
            temporal_ids = np.zeros(temporal * height * width, dtype=np.int64)
            height_ids = np.tile(np.repeat(np.arange(height, dtype=np.int64), width), temporal)
            width_ids = np.tile(np.arange(width, dtype=np.int64), temporal * height)
            sections.append(np.stack((temporal_ids, height_ids, width_ids)) + text_length + section_start)
            start = image_start + temporal * height * width
        if start < len(tokens):
            section_start = int(sections[-1].max()) + 1 if sections else 0
            text_length = len(tokens) - start
            sections.append(np.broadcast_to(np.arange(text_length, dtype=np.int64)[None], (3, text_length)).copy() + section_start)
        positions = np.concatenate(sections, axis=1)
        if positions.shape[1] != input_ids.shape[1]:
            raise ValueError(f"mRoPE length mismatch: positions={positions.shape[1]}, tokens={input_ids.shape[1]}")
        position_ids[:, batch_index] = positions
    if image_index != len(image_grid_thw):
        raise ValueError(f"mRoPE consumed {image_index} image grids, received {len(image_grid_thw)}")
    return position_ids


class System2Runner:
    def __init__(self, model_dir: Path, device: str, precision: str) -> None:
        ov_config = {"INFERENCE_PRECISION_HINT": precision}
        self.model = OVModelForVisualCausalLM.from_pretrained(model_dir, device=device, ov_config=ov_config, compile=False)
        language_graph = self.model.language_model.model
        logits = language_graph.output("logits")
        last_logits = ov.opset13.gather(
            logits.get_node().input_value(0),
            ov.opset13.constant([-1], dtype=np.int64),
            ov.opset13.constant(1, dtype=np.int64),
        )
        logits.get_node().input(0).replace_source_output(last_logits.output(0))
        language_graph.validate_nodes_and_infer_types()
        self.model.compile()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
        self.processor = AutoProcessor.from_pretrained(model_dir, use_fast=False)
        self.processor.tokenizer = self.tokenizer
        self.processor.tokenizer.padding_side = "left"
        self.latent_queries = np.load(model_dir / "latent_queries.npy")
        if self.latent_queries.shape != (1, TRAJECTORY_QUERY_COUNT, 3584):
            raise ValueError(f"invalid latent query shape: {self.latent_queries.shape}")

    def preprocess(self, source: dict[str, Any]) -> dict[str, np.ndarray]:
        images = [Image.open(path).convert("RGB").resize((384, 384)) for path in source["image_paths"]]
        content: list[dict[str, Any]] = []
        image_index = 0
        for part in split_prompt(source["prompt"]):
            if part == "<image>":
                content.append({"type": "image", "image": images[image_index]})
                image_index += 1
            else:
                content.append({"type": "text", "text": part})
        if image_index != len(images):
            raise ValueError(f"prompt consumed {image_index} images, received {len(images)}")
        return self.preprocess_messages([{"role": "user", "content": content}], images)

    def preprocess_messages(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
    ) -> dict[str, np.ndarray]:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processed = self.processor(text=[text], images=images, return_tensors="np")
        return {name: np.asarray(processed[name]) for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}

    def generate(
        self,
        inputs: dict[str, np.ndarray],
        max_new_tokens: int,
    ) -> tuple[np.ndarray, str, float]:
        torch_inputs = {name: torch.from_numpy(value) for name, value in inputs.items()}
        start = time.perf_counter()
        generated = self.model.generate(**torch_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed = time.perf_counter() - start
        generated_ids = generated.detach().cpu().numpy()
        continuation = generated_ids[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(continuation, skip_special_tokens=True)
        return generated_ids, text, elapsed

    def generate_trajectory_latents(
        self,
        generated_ids: np.ndarray,
        pixel_values: np.ndarray,
        image_grid_thw: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        input_ids = torch.from_numpy(generated_ids)
        grid = torch.from_numpy(image_grid_thw)
        text_embeddings = torch.as_tensor(self.model.get_text_embeddings(input_ids))
        image_embeddings = self.model.get_vision_embeddings(torch.from_numpy(pixel_values), grid_thw=grid)
        image_embeddings = torch.as_tensor(image_embeddings)
        image_mask = generated_ids == self.model.config.image_token_id
        if int(image_mask.sum()) != image_embeddings.shape[0]:
            raise ValueError("trajectory prefill image-token mismatch: " f"tokens={int(image_mask.sum())}, features={image_embeddings.shape[0]}")
        text_embeddings[torch.from_numpy(image_mask)] = image_embeddings.to(text_embeddings.dtype)
        inputs_embeds = torch.cat(
            (text_embeddings, torch.from_numpy(self.latent_queries).to(text_embeddings.dtype)),
            dim=1,
        )
        trajectory_tokens = np.full(
            (generated_ids.shape[0], TRAJECTORY_QUERY_COUNT),
            TRAJECTORY_TOKEN_ID,
            dtype=np.int64,
        )
        latent_input_ids = np.concatenate((generated_ids, trajectory_tokens), axis=1)
        vision_config = self.model.config.vision_config
        position_ids = qwen_image_rope_position_ids(
            latent_input_ids,
            image_grid_thw,
            image_token_id=self.model.config.image_token_id,
            vision_start_token_id=self.model.config.vision_start_token_id,
            spatial_merge_size=vision_config.spatial_merge_size,
        )
        request = self.model.language_model.request
        request.reset_state()
        try:
            request.infer(
                {
                    "inputs_embeds": inputs_embeds.detach().cpu().numpy(),
                    "attention_mask": np.ones(inputs_embeds.shape[:2], dtype=np.int64),
                    "position_ids": position_ids,
                    "beam_idx": np.zeros(generated_ids.shape[0], dtype=np.int32),
                },
                share_inputs=True,
            )
            latents = np.asarray(request.get_tensor("trajectory_latents").data).copy()
        finally:
            request.reset_state()
        elapsed = time.perf_counter() - start
        return latents, elapsed


class System1Runner:
    def __init__(self, model_dir: Path, device: str, precision: str) -> None:
        core = ov.Core()
        config = {"INFERENCE_PRECISION_HINT": precision}
        self.context = core.compile_model(model_dir / "s1_context_encoder.xml", device, config)
        self.denoiser = core.compile_model(model_dir / "s1_denoiser.xml", device, config)

    def generate_trajectories(
        self,
        trajectory_latents: np.ndarray,
        images: np.ndarray,
        initial_noise: np.ndarray,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if images.shape[-1] == 3:
            images = images.transpose(0, 1, 4, 2, 3)
        if images.shape != (1, 2, 3, 224, 224):
            raise ValueError(f"invalid System 1 image shape: {images.shape}")
        if trajectory_latents.shape != (1, TRAJECTORY_QUERY_COUNT, 3584):
            raise ValueError(f"invalid trajectory latent shape: {trajectory_latents.shape}")
        if initial_noise.ndim != 3 or initial_noise.shape[1:] != (32, 3):
            raise ValueError(f"invalid initial noise shape: {initial_noise.shape}")
        start = time.perf_counter()
        context = np.asarray(self.context({"images": images, "trajectory_latents": trajectory_latents})[self.context.output("context")])
        context_time = time.perf_counter() - start
        sample_count = initial_noise.shape[0]
        denoiser_context = np.concatenate((np.zeros_like(context), context), axis=0)
        denoiser_context = np.repeat(denoiser_context, sample_count, axis=0)
        trajectories = initial_noise.astype(np.float32, copy=True)
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float32)
        sigmas = np.concatenate((sigmas, np.zeros(1, dtype=np.float32)))
        denoiser_time = 0.0
        for index, sigma in enumerate(sigmas[:-1]):
            model_input = np.concatenate((trajectories, trajectories), axis=0)
            timestep = np.full(model_input.shape[0], round(sigma * 1000), dtype=np.int64)
            step_start = time.perf_counter()
            prediction = np.asarray(
                self.denoiser(
                    {
                        "trajectories": model_input,
                        "timestep": timestep,
                        "context": denoiser_context,
                    }
                )[self.denoiser.output("noise_prediction")]
            )
            denoiser_time += time.perf_counter() - step_start
            unconditional, conditional = np.split(prediction, 2)
            guided = unconditional + guidance_scale * (conditional - unconditional)
            trajectories = trajectories + (sigmas[index + 1] - sigma) * guided
        return trajectories, {
            "context_seconds": context_time,
            "denoiser_seconds": denoiser_time,
            "total_seconds": time.perf_counter() - start,
        }


def trajectories_to_actions(trajectories: np.ndarray) -> list[int]:
    deltas = trajectories.astype(np.float32, copy=True)
    deltas[:, :, :2] /= 4.0
    paths = np.zeros((len(deltas), deltas.shape[1] + 1, 2), dtype=np.float64)
    paths[:, 1:] = np.cumsum(deltas[:, :, :2], axis=1)
    trajectory = np.mean(paths, axis=0)
    actions: list[int] = []
    yaw = 0.0
    position = trajectory[0]
    goal = trajectory[-1]
    turn_angle = np.deg2rad(15)
    while np.linalg.norm(position - goal) > 0.2:
        nearest = int(np.argmin(np.linalg.norm(trajectory - position, axis=1)))
        target = trajectory[min(nearest + 4, len(trajectory) - 1)]
        direction = target - position
        if np.linalg.norm(direction) < 1e-6:
            break
        target_yaw = np.arctan2(direction[1], direction[0])
        delta_yaw = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        turns = int(round(delta_yaw / turn_angle))
        actions.extend(([2] if turns > 0 else [3]) * abs(turns))
        yaw = (yaw + turns * turn_angle + np.pi) % (2 * np.pi) - np.pi
        next_position = position + 0.25 * np.array([np.cos(yaw), np.sin(yaw)])
        if np.linalg.norm(next_position - goal) > np.linalg.norm(position - goal):
            break
        actions.append(1)
        position = next_position
    return actions
