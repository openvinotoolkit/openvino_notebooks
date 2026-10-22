"""Checkpoint loading utilities with no dependency on the InternNav source tree."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import GenerationConfig, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VisionRotaryEmbedding,
)

SYSTEM1_PREFIXES = (
    "model.action_decoder.",
    "model.action_encoder.",
    "model.cond_projector.",
    "model.latent_queries",
    "model.memory_encoder.",
    "model.rgb_model.",
    "model.rgb_resampler.",
    "model.traj_dit.",
)


def read_checkpoint_config(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("system1") != "nextdit_async":
        raise ValueError(f"unsupported System 1 variant: {config.get('system1')!r}")
    if config.get("n_query") != 4:
        raise ValueError(f"expected four trajectory queries, got {config.get('n_query')!r}")
    return config


def read_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]


def native_qwen_config(raw_config: dict[str, Any]) -> Qwen2_5_VLConfig:
    config = dict(raw_config)
    config["model_type"] = "qwen2_5_vl"
    config["architectures"] = ["Qwen2_5_VLForConditionalGeneration"]
    config["use_cache"] = True
    return Qwen2_5_VLConfig.from_dict(config)


def _native_qwen_key(checkpoint_key: str, native_keys: set[str]) -> str:
    if checkpoint_key in native_keys:
        return checkpoint_key
    if checkpoint_key.startswith("visual."):
        nested_visual = "model." + checkpoint_key
        if nested_visual in native_keys:
            return nested_visual
    if checkpoint_key.startswith("model."):
        nested = "model.language_model." + checkpoint_key.removeprefix("model.")
        if nested in native_keys:
            return nested
    return checkpoint_key


def validate_qwen_key_partition(model_dir: Path, config: Qwen2_5_VLConfig) -> None:
    checkpoint_keys = set(read_weight_map(model_dir))
    qwen_checkpoint_keys = {key for key in checkpoint_keys if not key.startswith(SYSTEM1_PREFIXES)}
    with torch.device("meta"):
        native_keys = set(Qwen2_5_VLForConditionalGeneration(config).state_dict())
    mapped_checkpoint_keys = {_native_qwen_key(key, native_keys) for key in qwen_checkpoint_keys}
    missing = sorted(native_keys - mapped_checkpoint_keys)
    unexpected = sorted(mapped_checkpoint_keys - native_keys)
    if missing or unexpected:
        raise RuntimeError("native Qwen checkpoint partition mismatch:\n" f"missing={missing[:20]}\nunexpected={unexpected[:20]}")
    required_custom = {
        "model.latent_queries",
        "model.action_encoder.weight",
        "model.action_decoder.weight",
        "model.cond_projector.0.weight",
        "model.rgb_model.patch_embed.proj.weight",
        "model.memory_encoder.memory_pos",
        "model.rgb_resampler.query_tokens",
        "model.traj_dit.model.patch_embedder.proj.weight",
    }
    absent_custom = sorted(required_custom - checkpoint_keys)
    if absent_custom:
        raise RuntimeError(f"checkpoint is missing required System 1 tensors: {absent_custom}")


def load_qwen_model(
    model_dir: Path,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Qwen2_5_VLForConditionalGeneration:
    raw_config = read_checkpoint_config(model_dir)
    config = native_qwen_config(raw_config)
    validate_qwen_key_partition(model_dir, config)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen2_5_VLForConditionalGeneration(config)
    model.to_empty(device=device)
    text_rotary = Qwen2_5_VLRotaryEmbedding(config.text_config, device=device)
    vision_head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
    vision_rotary = Qwen2_5_VisionRotaryEmbedding(vision_head_dim // 2).to(device)
    for module in model.modules():
        if isinstance(module, Qwen2_5_VLRotaryEmbedding):
            module.inv_freq = text_rotary.inv_freq.clone()
        elif isinstance(module, Qwen2_5_VisionRotaryEmbedding):
            module.inv_freq = vision_rotary.inv_freq.clone()
    native_state = model.state_dict()
    native_keys = set(native_state)
    weight_map = read_weight_map(model_dir)
    state_dict: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if not key.startswith(SYSTEM1_PREFIXES):
            by_shard.setdefault(shard, []).append(key)
    for shard, keys in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as tensors:
            for checkpoint_key in keys:
                native_key = _native_qwen_key(checkpoint_key, native_keys)
                tensor = tensors.get_tensor(checkpoint_key)
                if tuple(tensor.shape) != tuple(native_state[native_key].shape):
                    raise RuntimeError(
                        f"shape mismatch for {checkpoint_key}: checkpoint={tuple(tensor.shape)}, " f"native={tuple(native_state[native_key].shape)}"
                    )
                state_dict[native_key] = tensor
    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    # Upstream greedy decoding inherits repetition_penalty=1.05 and both EOS ids from this file.
    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    model.eval()
    model.to(dtype=dtype)
    return model


def load_module_prefix(
    module: torch.nn.Module,
    model_dir: Path,
    prefix: str,
    *,
    ignored_checkpoint_keys: tuple[str, ...] = (),
    allow_unexpected_checkpoint_keys: bool = False,
) -> None:
    """Load one checkpoint prefix into a standalone module with strict validation."""
    weight_map = read_weight_map(model_dir)
    selected = {key: shard for key, shard in weight_map.items() if key.startswith(prefix)}
    if not selected:
        raise KeyError(f"checkpoint contains no tensors with prefix {prefix!r}")

    state_dict: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for key, shard in selected.items():
        by_shard.setdefault(shard, []).append(key)
    for shard, keys in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as tensors:
            for key in keys:
                stripped = key.removeprefix(prefix)
                if stripped not in ignored_checkpoint_keys:
                    state_dict[stripped] = tensors.get_tensor(key)

    module_keys = set(module.state_dict())
    checkpoint_keys = set(state_dict)
    missing = sorted(module_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - module_keys)
    mismatched = sorted(key for key in module_keys & checkpoint_keys if tuple(module.state_dict()[key].shape) != tuple(state_dict[key].shape))
    if missing or (unexpected and not allow_unexpected_checkpoint_keys) or mismatched:
        raise RuntimeError(
            f"checkpoint prefix {prefix!r} does not match {type(module).__name__}:\n"
            f"missing={missing[:20]}\n"
            f"unexpected={unexpected[:20]}\n"
            f"mismatched={mismatched[:20]}"
        )
    if allow_unexpected_checkpoint_keys:
        state_dict = {key: value for key, value in state_dict.items() if key in module_keys}
    module.load_state_dict(state_dict, strict=True)
