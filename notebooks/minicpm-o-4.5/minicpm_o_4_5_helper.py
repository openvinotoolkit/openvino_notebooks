#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniCPM-o-4_5 OpenVINO Conversion and Inference Helper

This module provides:
1. Functions to convert all submodels of MiniCPM-o-4_5 duplex mode to OpenVINO format
2. OVMiniCPMO class for inference with simplex and duplex streaming modes
3. OVToken2wav for TTS audio synthesis

MiniCPM-o-4_5 Duplex Mode Submodels:
1. LLM (Language Model) - Qwen3ForCausalLM
2. LLM Embedding - Token embeddings
3. Vision Model (VPM) - SiglipVisionTransformer
4. Resampler - Visual resampler
5. Audio Model (APM) - MiniCPMWhisperEncoder
6. Audio Projection Layer - MultiModalProjector
7. TTS Model - MiniCPMTTS
   - TTS Text Embedding
   - TTS Model (LlamaModel backbone)
   - TTS Projector (Speaker & Semantic)
   - TTS Audio Code Embedding
   - TTS Audio Code Head
8. Token2wav - Flow Embeddings, Flow Estimator, HiFT vocoder

Conversion options:
    # Standard conversion (stateless APM)
    from minicpm_o_4_5_helper import convert_minicpmo_model
    convert_minicpmo_model("openbmb/MiniCPM-o-4_5", "./MiniCPM-o-4_5-OV")

    # Streaming conversion (APM with KV cache support - experimental)
    convert_minicpmo_model(
        "openbmb/MiniCPM-o-4_5",
        "./MiniCPM-o-4_5-OV-streaming",
        use_streaming_apm=True
    )

Usage example:
    from minicpm_o_4_5_helper import OVMiniCPMO

    # Load OpenVINO model
    model = OVMiniCPMO.from_pretrained(
        "/path/to/MiniCPM-o-4_5-OV",
        device="CPU"
    )

    # For duplex streaming mode
    model = model.as_duplex()
    model.prepare(prefix_system_prompt="Streaming Omni Conversation.")
    model.streaming_prefill(audio_waveform=audio_chunk, frame_list=frame_list)
    result = model.streaming_generate()
"""

import gc
import math
import os
import struct
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import asyncio

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    import openvino as ov
    from openvino import Core
    from openvino.frontend.pytorch.patch_model import __make_16bit_traceable

    try:
        from openvino import opset13
    except ImportError:
        from openvino.runtime import opset13
except ImportError:
    raise ImportError("Please install openvino: pip install openvino openvino-dev")

try:
    import nncf
except ImportError:
    nncf = None
    print("Warning: nncf not installed. Quantization will not be available.")


def _prepare_nncf_config(quantization_config: dict) -> dict:
    """Convert string mode values in quantization_config to nncf.CompressWeightsMode enum."""
    if quantization_config is None or nncf is None:
        return quantization_config
    config = dict(quantization_config)
    mode = config.get("mode")
    if isinstance(mode, str):
        mode_map = {
            "int4_sym": nncf.CompressWeightsMode.INT4_SYM,
            "int4_asym": nncf.CompressWeightsMode.INT4_ASYM,
            "int8_sym": nncf.CompressWeightsMode.INT8_SYM,
            "int8_asym": nncf.CompressWeightsMode.INT8_ASYM,
            "int8": nncf.CompressWeightsMode.INT8_SYM,
        }
        config["mode"] = mode_map.get(mode.lower(), nncf.CompressWeightsMode.INT8_SYM)
    return config


from transformers import AutoModel, AutoConfig, AutoTokenizer, GenerationConfig
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

# OpenVINO core
core = Core()

# File naming conventions for MiniCPM-o-4_5
LLM_EMBEDDING_NAME = "openvino_llm_embedding_model.xml"
LLM_LANGUAGE_NAME = "openvino_llm_language_model.xml"
VISION_NAME = "openvino_vision_model.xml"
RESAMPLER_NAME = "openvino_resampler_model.xml"
AUDIO_ENCODER_NAME = "openvino_audio_encoder_model.xml"
AUDIO_PROJECTION_NAME = "openvino_audio_projection_model.xml"
TTS_EMBEDDING_NAME = "openvino_tts_embedding_model.xml"
TTS_LANGUAGE_NAME = "openvino_tts_language_model.xml"
TTS_PROJECTOR_SPK_NAME = "openvino_tts_projector_spk_model.xml"
TTS_PROJECTOR_SEMANTIC_NAME = "openvino_tts_projector_semantic_model.xml"
TTS_CODE_EMBEDDING_NAME = "openvino_tts_code_embedding_model.xml"
TTS_CODE_HEAD_NAME = "openvino_tts_code_head_model.xml"

# Token2wav (audio_tokenizer) model names
FLOW_EMBEDDINGS_NAME = "openvino_flow_embeddings_model.xml"
FLOW_ESTIMATOR_NAME = "openvino_flow_estimator_model.xml"
HIFT_NAME = "openvino_hift_model.xml"


def cleanup_torchscript_cache():
    """Helper for removing cached model representation."""
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def _strip_hf_repo_prefix(value: str) -> str:
    """
    Strip 'owner/repo--' prefix from a HuggingFace auto_map class reference.

    E.g. 'openbmb/MiniCPM-o-4_5--processing_minicpmo.MiniCPMVImageProcessor'
    becomes 'processing_minicpmo.MiniCPMVImageProcessor'.
    Returns the original string unchanged if it doesn't match the pattern.
    """
    if isinstance(value, str) and "/" in value and "--" in value:
        prefix, sep, module_class = value.partition("--")
        if sep and "/" in prefix:  # looks like owner/repo
            return module_class
    return value


def _patch_auto_map_in_dir(output_path: Path):
    """
    Strip HuggingFace repo prefix from auto_map entries in all JSON config files.

    Saved configs like preprocessor_config.json, config.json, and tokenizer_config.json
    may contain auto_map entries referencing the original HF repo, e.g.:
        "AutoImageProcessor": "openbmb/MiniCPM-o-4_5--processing_minicpmo.MiniCPMVImageProcessor"
    These are patched to local-only format:
        "AutoImageProcessor": "processing_minicpmo.MiniCPMVImageProcessor"
    so that transformers loads custom classes from local files without downloading from HF.
    List values (e.g. AutoTokenizer) are handled element-by-element.
    """
    import json

    for json_file in output_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "auto_map" not in data:
                continue
            patched = False
            for key, value in data["auto_map"].items():
                if isinstance(value, str):
                    new_val = _strip_hf_repo_prefix(value)
                    if new_val != value:
                        data["auto_map"][key] = new_val
                        patched = True
                elif isinstance(value, list):
                    new_list = [_strip_hf_repo_prefix(v) if isinstance(v, str) else v for v in value]
                    if new_list != value:
                        data["auto_map"][key] = new_list
                        patched = True
            if patched:
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"✅ Patched auto_map in {json_file.name}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to patch {json_file.name}: {e}")


def _copy_custom_code_files(model_id: str, output_path: Path):
    """
    Copy custom Python code files to the output directory for fully offline use.

    Transformers loads custom classes (AutoImageProcessor, AutoTokenizer, etc.)
    from .py files referenced in auto_map. For offline operation these files must
    reside in the model directory itself.

    - Local model_id: copies all *.py directly from the source folder.
    - HF Hub model_id: downloads only the *.py files via snapshot_download.
    """
    import shutil

    src_path = Path(model_id)
    if src_path.exists() and src_path.is_dir():
        py_files = sorted(src_path.glob("*.py"))
        copied = 0
        for f in py_files:
            dst = output_path / f.name
            shutil.copy2(f, dst)
            copied += 1
        if copied:
            print(f"✅ Copied {copied} custom code .py files")
    else:
        try:
            from huggingface_hub import snapshot_download

            cache_dir = Path(snapshot_download(model_id, allow_patterns=["*.py"]))
            py_files = sorted(cache_dir.glob("*.py"))
            copied = 0
            for f in py_files:
                dst = output_path / f.name
                shutil.copy2(f, dst)
                copied += 1
            if copied:
                print(f"✅ Copied {copied} custom code .py files from Hub cache")
        except Exception as e:
            print(f"⚠️ Warning: Failed to copy custom code files: {e}")


def patch_cos_sin_cached_fp32(model):
    """Patch rotary embeddings to use FP32 for better accuracy."""
    if (
        hasattr(model, "layers")
        and hasattr(model.layers[0], "self_attn")
        and hasattr(model.layers[0].self_attn, "rotary_emb")
        and hasattr(model.layers[0].self_attn.rotary_emb, "dtype")
        and hasattr(model.layers[0].self_attn.rotary_emb, "inv_freq")
        and hasattr(model.layers[0].self_attn.rotary_emb, "max_position_embeddings")
        and hasattr(model.layers[0].self_attn.rotary_emb, "_set_cos_sin_cache")
    ):
        for layer in model.layers:
            if layer.self_attn.rotary_emb.dtype != torch.float32:
                layer.self_attn.rotary_emb._set_cos_sin_cache(
                    seq_len=layer.self_attn.rotary_emb.max_position_embeddings,
                    device=layer.self_attn.rotary_emb.inv_freq.device,
                    dtype=torch.float32,
                )


def model_has_state(ov_model: ov.Model):
    """Check if OpenVINO model has states (stateful model)."""
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """Helper function for checking that model has specified input or output name."""
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: list,
    key_value_input_names: list,
    gather_dim: int,
):
    """
    Fuses reordered cache during generate cycle into ov.Model.
    Used with stateful models for beam search support.
    """
    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])

    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """Build initialization ShapeOf Expression for all ReadValue ops."""
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: list,
    key_value_input_names: list,
    key_value_output_names: list,
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """Hides kv-cache inputs and outputs inside the model as variables."""
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)

    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model, dim=1):
    """Apply stateful transformation to OpenVINO model."""
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs[2:-1]]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs[dim:]]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )


def convert_minicpmo_model(
    model_id: str,
    output_dir: str,
    quantization_config: dict = None,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float32,
    use_streaming_apm: bool = False,
    use_stateful_apm: bool = False,
):
    """
    Convert MiniCPM-o-4_5 model to OpenVINO format.

    This function converts all 12 submodels used in duplex mode:
    1. LLM Embedding
    2. LLM Language Model
    3. Vision Model (VPM)
    4. Resampler
    5. Audio Encoder (APM)
    6. Audio Projection Layer
    7. TTS Text Embedding
    8. TTS Language Model
    9. TTS Projector (Speaker)
    10. TTS Projector (Semantic)
    11. TTS Audio Code Embedding
    12. TTS Audio Code Head

    Args:
        model_id: Model identifier or path to local model
        output_dir: Directory to save converted models
        quantization_config: Optional NNCF quantization configuration
        device: Device for model loading ("cpu" or "cuda")
        torch_dtype: Torch dtype for model loading
        use_streaming_apm: If True, convert APM with KV cache support for streaming (experimental)
        use_stateful_apm: If True, convert APM with stateful KV cache (KV cache hidden as internal state)

    Returns:
        Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract vision and llm specific quantization configs
    vision_quantization_config = None
    llm_quantization_config = None
    if quantization_config is not None:
        if isinstance(quantization_config, dict):
            vision_quantization_config = quantization_config.get("vision")
            llm_quantization_config = quantization_config.get("llm")
            text_llm_quantization_config = quantization_config.get("text")
        else:
            # Fallback: if quantization_config is not a dict with keys, use it for both
            vision_quantization_config = quantization_config
            llm_quantization_config = quantization_config
            text_llm_quantization_config = quantization_config

    # Define all model paths
    llm_embedding_path = output_path / LLM_EMBEDDING_NAME
    llm_language_path = output_path / LLM_LANGUAGE_NAME
    vision_path = output_path / VISION_NAME
    resampler_path = output_path / RESAMPLER_NAME
    audio_encoder_path = output_path / AUDIO_ENCODER_NAME
    audio_projection_path = output_path / AUDIO_PROJECTION_NAME
    tts_embedding_path = output_path / TTS_EMBEDDING_NAME
    tts_language_path = output_path / TTS_LANGUAGE_NAME
    tts_projector_spk_path = output_path / TTS_PROJECTOR_SPK_NAME
    tts_projector_semantic_path = output_path / TTS_PROJECTOR_SEMANTIC_NAME
    tts_code_embedding_path = output_path / TTS_CODE_EMBEDDING_NAME
    tts_code_head_path = output_path / TTS_CODE_HEAD_NAME

    # Check if all models already exist
    all_paths = [
        llm_embedding_path,
        llm_language_path,
        vision_path,
        resampler_path,
        audio_encoder_path,
        audio_projection_path,
        tts_embedding_path,
        tts_language_path,
        tts_projector_spk_path,
        tts_projector_semantic_path,
        tts_code_embedding_path,
        tts_code_head_path,
    ]

    if all(p.exists() for p in all_paths):
        print(f"✅ {model_id} model already converted. You can find results in {output_dir}")
        # Ensure generation_config.json exists in the output directory
        if not (output_path / "generation_config.json").exists():
            try:
                GenerationConfig.from_pretrained(model_id, trust_remote_code=True).save_pretrained(str(output_path))
                print("✅ generation_config.json saved")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save generation_config: {e}")
        # Copy custom .py code files for offline loading
        _copy_custom_code_files(model_id, output_path)
        # Patch auto_map to use local paths
        _patch_auto_map_in_dir(output_path)
        return output_path

    print(f"⌛ {model_id} conversion started. Be patient, it may take some time.")
    print("⌛ Loading Original model...")

    # Load the model
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch_dtype,
    )
    model.eval()
    model.to(device)

    # Save config, tokenizer, and processor
    model.config.save_pretrained(output_dir)

    # Save tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        print("✅ Tokenizer saved")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save tokenizer: {e}")

    # Save processor
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        processor.save_pretrained(output_dir)
        print("✅ Processor saved")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save processor: {e}")

    # Save generation_config.json
    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(output_dir)
        else:
            GenerationConfig.from_pretrained(model_id, trust_remote_code=True).save_pretrained(output_dir)
        print("✅ generation_config.json saved")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save generation_config: {e}")

    # Copy custom .py code files needed for offline loading
    _copy_custom_code_files(model_id, output_path)
    # Patch auto_map in all saved JSON configs to use local paths (not HF repo refs)
    _patch_auto_map_in_dir(output_path)

    print("✅ Original model successfully loaded")

    # 1. Convert LLM Embedding
    if not llm_embedding_path.exists():
        print("⌛ Converting LLM Embedding model...")
        convert_llm_embedding(model, llm_embedding_path, text_llm_quantization_config)
        print("✅ LLM Embedding model successfully converted")

    # 2. Convert LLM Language Model
    if not llm_language_path.exists():
        print("⌛ Converting LLM Language model...")
        convert_llm_language_model(model, llm_language_path, llm_quantization_config)
        print("✅ LLM Language model successfully converted")

    # 3. Convert Vision Model (VPM)
    if not vision_path.exists() and model.config.init_vision:
        print("⌛ Converting Vision model (VPM)...")
        convert_vision_model(model, vision_path, vision_quantization_config)
        print("✅ Vision model successfully converted")

    # 4. Convert Resampler
    if not resampler_path.exists() and model.config.init_vision:
        print("⌛ Converting Resampler model...")
        convert_resampler(model, resampler_path)
        print("✅ Resampler model successfully converted")

    # 5. Convert Audio Encoder (APM)
    if not audio_encoder_path.exists() and model.config.init_audio:
        if use_stateful_apm:
            print("⌛ Converting Audio Encoder model (APM) with stateful KV cache...")
            convert_audio_encoder_stateful(model, audio_encoder_path)
            print("✅ Audio Encoder model (stateful) successfully converted")
        elif use_streaming_apm:
            print("⌛ Converting Audio Encoder model (APM) with streaming support...")
            convert_audio_encoder_streaming(model, audio_encoder_path)
            print("✅ Audio Encoder model (streaming) successfully converted")
        else:
            print("⌛ Converting Audio Encoder model (APM)...")
            convert_audio_encoder(model, audio_encoder_path)
            print("✅ Audio Encoder model successfully converted")

    # 6. Convert Audio Projection Layer
    if not audio_projection_path.exists() and model.config.init_audio:
        print("⌛ Converting Audio Projection model...")
        convert_audio_projection(model, audio_projection_path)
        print("✅ Audio Projection model successfully converted")

    # TTS Model Conversion
    if model.config.init_tts:
        # 7. Convert TTS Text Embedding
        if not tts_embedding_path.exists():
            print("⌛ Converting TTS Text Embedding model...")
            convert_tts_text_embedding(model, tts_embedding_path)
            print("✅ TTS Text Embedding model successfully converted")

        # 8. Convert TTS Language Model (No quantization for TTS)
        if not tts_language_path.exists():
            print("⌛ Converting TTS Language model...")
            convert_tts_language_model(model, tts_language_path, quantization_config=None)
            print("✅ TTS Language model successfully converted")

        # 9. Convert TTS Projector (Speaker)
        if not tts_projector_spk_path.exists():
            print("⌛ Converting TTS Projector (Speaker) model...")
            convert_tts_projector_spk(model, tts_projector_spk_path)
            print("✅ TTS Projector (Speaker) model successfully converted")

        # 10. Convert TTS Projector (Semantic)
        if not tts_projector_semantic_path.exists():
            print("⌛ Converting TTS Projector (Semantic) model...")
            convert_tts_projector_semantic(model, tts_projector_semantic_path)
            print("✅ TTS Projector (Semantic) model successfully converted")

        # 11. Convert TTS Audio Code Embedding
        if not tts_code_embedding_path.exists():
            print("⌛ Converting TTS Audio Code Embedding model...")
            convert_tts_code_embedding(model, tts_code_embedding_path)
            print("✅ TTS Audio Code Embedding model successfully converted")

        # 12. Convert TTS Audio Code Head
        if not tts_code_head_path.exists():
            print("⌛ Converting TTS Audio Code Head model...")
            convert_tts_code_head(model, tts_code_head_path)
            print("✅ TTS Audio Code Head model successfully converted")

    # 13-15. Convert Token2wav models (Flow Embeddings, Flow Estimator, HiFT)
    # Check if audio_tokenizer path is available in model config
    audio_tokenizer_path = None
    if hasattr(model.config, "audio_tokenizer_path") and model.config.audio_tokenizer_path:
        audio_tokenizer_path = model.config.audio_tokenizer_path
        print(f"⌛ Found audio_tokenizer_path in config: {audio_tokenizer_path}")
    elif model.config.init_tts:
        # Try local path first
        local_model_dir = Path(model_id) if Path(model_id).exists() else None
        if local_model_dir:
            token2wav_dir = local_model_dir / "assets" / "token2wav"
            if token2wav_dir.exists() and (token2wav_dir / "flow.yaml").exists():
                audio_tokenizer_path = str(token2wav_dir)
                print(f"⌛ Found audio_tokenizer at: {audio_tokenizer_path}")
        else:
            # model_id is a HuggingFace Hub repo ID — download only assets/token2wav/
            try:
                from huggingface_hub import snapshot_download

                print(f"⌛ Downloading assets/token2wav from Hub repo: {model_id} ...")
                hub_cache_dir = snapshot_download(
                    repo_id=model_id,
                    allow_patterns=["assets/token2wav/**"],
                )
                token2wav_dir = Path(hub_cache_dir) / "assets" / "token2wav"
                if token2wav_dir.exists() and (token2wav_dir / "flow.yaml").exists():
                    audio_tokenizer_path = str(token2wav_dir)
                    print(f"⌛ Found audio_tokenizer at: {audio_tokenizer_path}")
                else:
                    print("⚠️ assets/token2wav not found in Hub repo snapshot")
            except Exception as e:
                print(f"⚠️ Warning: Could not download audio_tokenizer from Hub: {e}")

    if audio_tokenizer_path and model.config.init_tts:
        try:
            print("⌛ Converting Token2wav models (Flow Embeddings, Flow Estimator, HiFT)...")
            convert_token2wav(
                audio_tokenizer_path=audio_tokenizer_path,
                output_dir=str(output_path),
                float16=(torch_dtype == torch.float16),
            )
            print("✅ Token2wav models successfully converted")
        except Exception as e:
            print(f"⚠️ Warning: Failed to convert Token2wav models: {e}")
            print("   You may need to convert Token2wav separately using convert_token2wav.py")
    elif model.config.init_tts:
        print("⚠️ Warning: TTS is enabled but audio_tokenizer_path not found")
        print("   Please convert Token2wav separately using convert_token2wav.py")
        print("   Example: python convert_token2wav.py --audio-tokenizer-path <path> --output-dir <output>")

    gc.collect()

    total_models = 12
    if audio_tokenizer_path and model.config.init_tts:
        total_models = 15  # Include Flow Embeddings, Flow Estimator, HiFT
    del model

    print(f"✅ {model_id} model conversion finished ({total_models} models). Results in {output_dir}")
    return output_path


def convert_llm_embedding(model, output_path: Path, quantization_config=None):
    """Convert LLM embedding layer to OpenVINO."""
    embed_tokens = model.llm.model.embed_tokens
    __make_16bit_traceable(embed_tokens)

    ov_model = ov.convert_model(
        embed_tokens,
        example_input=torch.ones([1, 10], dtype=torch.int64),
    )
    if quantization_config is not None and nncf is not None:
        print(f"⌛ Weights compression with {quantization_config.get('mode', 'int8')} mode started")
        ov_model = nncf.compress_weights(ov_model, **_prepare_nncf_config(quantization_config))
        print("✅ Weights compression finished")
    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def convert_llm_language_model(model, output_path: Path, quantization_config=None):
    """Convert LLM language model to OpenVINO with stateful KV cache."""

    def forward_wrap_llm(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
    ):
        if past_key_values is not None:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        if past_key_values is not None:
            outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # Return logits, last hidden states, and KV cache
        return (logits, hidden_states, outputs.past_key_values)

    llm = model.llm
    hidden_size = llm.config.hidden_size

    patch_cos_sin_cached_fp32(llm)
    if hasattr(llm, "model"):
        patch_cos_sin_cached_fp32(llm.model)

    llm._orig_forward = llm.forward
    llm.forward = types.MethodType(forward_wrap_llm, llm)

    num_pkv = llm.config.num_hidden_layers
    num_kv_heads = llm.config.num_key_value_heads
    head_dim = llm.config.head_dim if hasattr(llm.config, "head_dim") else (hidden_size // llm.config.num_attention_heads)

    pkv_shape = (2, num_kv_heads, 2, head_dim)

    # Use float32 for tracing to avoid dtype mismatches in attention
    inputs_embeds = torch.randn((2, 2, hidden_size), dtype=torch.float32)
    attention_mask = torch.ones([2, 4], dtype=torch.int64)
    position_ids = torch.arange(2, 4).unsqueeze(0).expand(2, -1)

    input_names = ["attention_mask", "position_ids"]
    output_names = ["logits", "hidden_states"]
    past_key_values = []

    for i in range(num_pkv):
        kv = [torch.randn(pkv_shape, dtype=torch.float32) for _ in range(2)]
        past_key_values.append(kv)
        input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names.extend([f"present.{i}.key", f"present.{i}.value"])
    input_names.append("inputs_embeds")

    example_input = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    }

    input_shapes = [
        ov.PartialShape([-1, -1]),  # attention_mask
        ov.PartialShape([-1, -1]),  # position_ids
    ]
    input_shapes += [ov.PartialShape([-1, num_kv_heads, -1, head_dim])] * 2 * num_pkv
    input_shapes += [ov.PartialShape([-1, -1, hidden_size])]  # inputs_embeds

    __make_16bit_traceable(llm)

    ov_model = ov.convert_model(llm, example_input=example_input, input=input_shapes)

    for input, input_name in zip(ov_model.inputs, input_names):
        input.get_tensor().set_names({input_name})

    for output, output_name in zip(ov_model.outputs, output_names):
        output.get_tensor().set_names({output_name})

    # Apply stateful transformation: KV cache is hidden inside the model as internal variables.
    # This eliminates the need to pass/receive KV cache tensors at each infer() call.
    # dim=2 because the first 2 outputs (logits, hidden_states) are NOT KV cache.
    patch_stateful(ov_model, 2)
    print("✅ LLM Language Model converted to stateful mode")

    if quantization_config is not None and nncf is not None:
        print(f"⌛ Weights compression with {quantization_config.get('mode', 'int8')} mode started")
        ov_model = nncf.compress_weights(ov_model, **_prepare_nncf_config(quantization_config))
        print("✅ Weights compression finished")

    ov.save_model(ov_model, output_path)

    llm.forward = llm._orig_forward
    del llm._orig_forward
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def convert_vision_model(model, output_path: Path, quantization_config=None):
    """Convert Vision model (SiglipVisionTransformer) to OpenVINO.

    The original SiglipVisionEmbeddings.forward uses a for-loop with boolean mask indexing
    (position_ids[batch_idx][mask] = pos_ids) that traces to ScatterNDUpdate with fixed shapes,
    which fails at inference time with different image sizes.

    Solution: accept pre-computed position_ids as input, bypass the problematic embeddings
    forward, and do patch_embed + position_embed directly.
    """
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

    vpm = model.vpm

    def forward_wrap_vision(
        self,
        pixel_values,
        patch_attention_mask=None,
        position_ids=None,
    ):
        # Bypass self.embeddings.forward to avoid the problematic index_put_ operation.
        # Instead, do patch embedding + position embedding directly with pre-computed position_ids.
        patch_embeds = self.embeddings.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.embeddings.position_embedding(position_ids)
        hidden_states = embeddings

        batch_size = pixel_values.shape[0]
        patch_attention_mask = patch_attention_mask.view(batch_size, -1)

        # Align with original model: skip attention_mask if all 1s
        if not torch.any(~patch_attention_mask):
            attention_mask = None
        else:
            attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

    vpm._orig_forward = vpm.forward
    vpm.forward = types.MethodType(forward_wrap_vision, vpm)

    __make_16bit_traceable(vpm)

    # Example input with dynamic shape support
    batch_size = 1
    channels = 3
    patch_size = vpm.embeddings.patch_size
    num_patches_h = 14
    num_patches_w = 14
    num_patches = num_patches_h * num_patches_w  # 196 patches

    pixel_values = torch.randn(
        [batch_size, channels, patch_size * num_patches_h, patch_size * num_patches_w],
        dtype=vpm.embeddings.patch_embedding.weight.dtype,
    )
    patch_attention_mask = torch.ones([batch_size, 1, num_patches], dtype=torch.bool)
    position_ids = torch.arange(num_patches, dtype=torch.long).unsqueeze(0)  # [1, num_patches]

    # Input shapes: fix channels=3, allow dynamic batch, height, width
    input_shapes = [
        ov.PartialShape([-1, 3, -1, -1]),  # pixel_values: [batch, channels=3, H, W]
        ov.PartialShape([-1, 1, -1]),  # patch_attention_mask: [batch, 1, num_patches]
        ov.PartialShape([-1, -1]),  # position_ids: [batch, num_patches]
    ]

    ov_model = ov.convert_model(
        vpm,
        example_input={
            "pixel_values": pixel_values,
            "patch_attention_mask": patch_attention_mask,
            "position_ids": position_ids,
        },
        input=input_shapes,
    )

    if quantization_config is not None and nncf is not None:
        print(f"⌛ Weights compression with {quantization_config.get('mode', 'int8')} mode started")
        ov_model = nncf.compress_weights(ov_model, **_prepare_nncf_config(quantization_config))
        print("✅ Weights compression finished")

    ov.save_model(ov_model, output_path)

    vpm.forward = vpm._orig_forward
    del vpm._orig_forward
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def convert_resampler(model, output_path: Path, quantization_config=None):
    """Convert Resampler to OpenVINO - only core operations with weights."""
    resampler = model.resampler

    def forward_wrap_resampler(self, x, pos_embed):
        """
        Simplified wrapper with only weighted operations.
        Args:
            x: visual features [B, L, vision_dim]
            pos_embed: position embeddings [L, B, embed_dim] (precomputed in pipeline)
        """
        # Core operations with weights only
        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query).unsqueeze(1)  # Q * 1 * D

        out = self.attn(
            q,  # Q * 1 * D
            x + pos_embed,  # L * 1 * D
            x,
            key_padding_mask=None,
        )[0]
        #  out: Q * 1 * D
        x = out.permute(1, 0, 2)  # 1 * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x

    resampler._orig_forward = resampler.forward
    resampler.forward = types.MethodType(forward_wrap_resampler, resampler)

    __make_16bit_traceable(resampler)

    # Get dimensions
    vision_dim = model.vision_dim
    embed_dim = model.embed_dim
    num_patches = 196  # 14x14

    # Example inputs
    visual_features = torch.randn([1, num_patches, vision_dim], dtype=resampler.proj.dtype)
    pos_embed = torch.randn([num_patches, 1, embed_dim], dtype=resampler.proj.dtype)

    # Input shapes: fix vision_dim and embed_dim, allow dynamic seq_len
    input_shapes = [
        ov.PartialShape([1, -1, vision_dim]),  # x: [1, num_patches, vision_dim]
        ov.PartialShape([-1, 1, embed_dim]),  # pos_embed: [num_patches, 1, embed_dim]
    ]

    ov_model = ov.convert_model(
        resampler,
        example_input={
            "x": visual_features,
            "pos_embed": pos_embed,
        },
        input=input_shapes,
    )

    if quantization_config is not None and nncf is not None:
        print(f"⌛ Weights compression with {quantization_config.get('mode', 'int8')} mode started")
        ov_model = nncf.compress_weights(ov_model, **_prepare_nncf_config(quantization_config))
        print("✅ Weights compression finished")

    ov.save_model(ov_model, output_path)

    resampler.forward = resampler._orig_forward
    del resampler._orig_forward
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def convert_audio_encoder(model, output_path: Path):
    """Convert Audio Encoder (MiniCPMWhisperEncoder) to OpenVINO."""
    apm = model.apm

    def forward_wrap_audio(
        self,
        input_features,
        attention_mask=None,
    ):
        # CNN feature extraction
        inputs_embeds = F.gelu(self.conv1(input_features))
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight[: inputs_embeds.shape[1]]
        embed_pos = embed_pos[: inputs_embeds.shape[1], :]
        hidden_states = inputs_embeds + embed_pos
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # Encoder layers
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                layer_head_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states

    apm._orig_forward = apm.forward
    apm.forward = types.MethodType(forward_wrap_audio, apm)

    __make_16bit_traceable(apm)

    # Example input: mel spectrogram
    batch_size = 1
    mel_length = 3000  # ~30 seconds of audio
    mel_bins = 80

    input_features = torch.randn([batch_size, mel_bins, mel_length], dtype=apm.conv1.weight.dtype)

    # Input shapes: fix mel_bins=80, allow dynamic batch and mel_length
    input_shapes = [
        ov.PartialShape([-1, mel_bins, -1]),  # input_features: [batch, mel_bins=80, mel_length]
    ]

    ov_model = ov.convert_model(
        apm,
        example_input={"input_features": input_features},
        input=input_shapes,
    )

    ov.save_model(ov_model, output_path)

    apm.forward = apm._orig_forward
    del apm._orig_forward
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def convert_audio_encoder_streaming(model, output_path: Path):
    """Convert Audio Encoder (MiniCPMWhisperEncoder) to OpenVINO with KV cache support.

    MiniCPMWhisperEncoder is a modified Whisper encoder that supports KV cache through
    EncoderDecoderCache structure, enabling efficient streaming audio processing.
    """
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    apm = model.apm

    def forward_wrap_audio_streaming(
        self,
        input_features,
        attention_mask=None,
        past_key_values=None,
    ):
        """Forward pass with KV cache support using EncoderDecoderCache.

        This matches MiniCPMWhisperEncoder.forward logic for cache handling.
        """
        # CNN feature extraction (same as original)
        inputs_embeds = F.gelu(self.conv1(input_features))
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        # Position embeddings with cache handling (aligned with original lines 3963-3986)
        embed_pos = self.embed_positions.weight
        past_key_values_length = 0
        use_cache = True  # Always use cache during conversion

        if use_cache:
            # Initialize or convert cache to EncoderDecoderCache
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, tuple):
                past_key_values = EncoderDecoderCache(DynamicCache.from_legacy_cache(past_key_values), DynamicCache())
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())

            # Get cached sequence length
            past_key_values_length = past_key_values.self_attention_cache.get_usable_length(inputs_embeds.shape[1])

            # Handle position embeddings with cache offset
            if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:
                # Extend position embeddings if needed
                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat(
                    (
                        embed_pos_front,
                        torch.repeat_interleave(
                            embed_pos[-1, :].unsqueeze(0),
                            inputs_embeds.shape[1] - embed_pos.shape[0] + past_key_values_length,
                            dim=0,
                        ),
                    )
                )
            else:
                embed_pos = embed_pos[past_key_values_length : inputs_embeds.shape[1] + past_key_values_length, :]
        else:
            embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # Encoder layers with KV cache (aligned with original lines 4007-4034)
        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                layer_head_mask=None,
                past_key_values=past_key_values,  # Pass EncoderDecoderCache
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            # Update cache from layer output
            if use_cache:
                next_encoder_cache = layer_outputs[1]  # index 1 when output_attentions=False

        hidden_states = self.layer_norm(hidden_states)
        if past_key_values is not None:
            next_encoder_cache = next_encoder_cache.to_legacy_cache()
        # Return hidden states and updated cache
        return (hidden_states, next_encoder_cache)

    apm._orig_forward = apm.forward
    apm.forward = types.MethodType(forward_wrap_audio_streaming, apm)

    __make_16bit_traceable(apm)

    # Example input: mel spectrogram for streaming chunks
    batch_size = 1
    mel_length = 300  # ~3 seconds per chunk for streaming
    mel_bins = 80

    input_features = torch.randn([batch_size, mel_bins, mel_length], dtype=torch.float32)

    # Calculate sequence length after CNN (stride=2 in conv2)
    seq_len_after_cnn = (mel_length - 1) // 2 + 1  # 150

    # Construct KV cache inputs (similar to LLM pattern)
    input_names = ["input_features", "attention_mask"]
    output_names = ["hidden_states"]
    past_key_values = []

    # Get APM configuration for cache construction
    num_layers = apm.config.encoder_layers
    num_heads = apm.config.encoder_attention_heads
    hidden_size = apm.config.d_model
    head_dim = hidden_size // num_heads

    # Create past key values with seq_len=2 (simulating cached frames)
    past_seq_len = 2
    for i in range(num_layers):
        # Each layer has [key, value] pair
        key = torch.randn(batch_size, num_heads, past_seq_len, head_dim, dtype=torch.float32)
        value = torch.randn(batch_size, num_heads, past_seq_len, head_dim, dtype=torch.float32)
        past_key_values.append([key, value])
        input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names.extend([f"present.{i}.key", f"present.{i}.value"])

    # Attention mask shape: [batch, 1, current_seq_len, total_seq_len]
    # total_seq_len = past_seq_len + current_seq_len
    total_seq_len = past_seq_len + seq_len_after_cnn  # 2 + 150 = 152
    attention_mask = torch.zeros([batch_size, 1, seq_len_after_cnn, total_seq_len], dtype=torch.float32)

    example_input = {
        "input_features": input_features,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
    }

    # Define input shapes with dynamic dimensions
    input_shapes = [
        ov.PartialShape([-1, mel_bins, -1]),  # input_features: [batch, 80, mel_length]
        ov.PartialShape([-1, 1, -1, -1]),  # attention_mask: [batch, 1, seq_len, total_seq_len]
    ]
    # Add KV cache shapes: [batch, num_heads, seq_len, head_dim]
    input_shapes += [ov.PartialShape([-1, num_heads, -1, head_dim])] * 2 * num_layers

    # Convert model with KV cache
    ov_model = ov.convert_model(
        apm,
        example_input=example_input,
        input=input_shapes,
    )

    # Set tensor names for inputs and outputs
    for input, input_name in zip(ov_model.inputs, input_names):
        input.get_tensor().set_names({input_name})

    for output, output_name in zip(ov_model.outputs, output_names):
        output.get_tensor().set_names({output_name})

    ov.save_model(ov_model, output_path)

    apm.forward = apm._orig_forward
    del apm._orig_forward
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()

    print(f"✅ Audio Encoder model (streaming-ready) saved to {output_path}")
    print(f"ℹ️  Model accepts attention_mask input for future cache support")


def build_state_initializer_apm(ov_model: ov.Model, batch_dim: int):
    """Build initialization ShapeOf Expression for all ReadValue ops in APM model.

    Similar to build_state_initializer but uses 'input_features' instead of 'inputs_embeds'.
    """
    input_features = ov_model.input("input_features")
    batch = opset13.gather(
        opset13.shape_of(input_features, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful_apm(
    ov_model: ov.Model,
    not_kv_inputs: list,
    key_value_input_names: list,
    key_value_output_names: list,
    batch_dim: int,
):
    """Hides kv-cache inputs and outputs inside the APM model as variables.

    Similar to make_stateful but uses build_state_initializer_apm and adds beam_idx.
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]

    apply_make_stateful_transformation(ov_model, input_output_map)
    build_state_initializer_apm(ov_model, batch_dim)


def patch_stateful_apm(ov_model):
    """Apply stateful transformation to APM OpenVINO model.

    Unlike patch_stateful for LLM, this handles the APM's input/output layout:
    - Inputs: input_features, attention_mask, past_key_values.0.key, ..., past_key_values.N.value
    - Outputs: hidden_states, present.0.key, ..., present.N.value

    The KV cache inputs are all inputs after the first 2 (input_features, attention_mask).
    The KV cache outputs are all outputs after the first 1 (hidden_states).
    No beam_idx is added (APM doesn't need beam search reordering).
    """
    # Identify KV cache input/output names
    all_input_names = [inp.get_any_name() for inp in ov_model.inputs]
    all_output_names = [out.get_any_name() for out in ov_model.outputs]

    key_value_input_names = [name for name in all_input_names if "past_key_values" in name]
    key_value_output_names = [name for name in all_output_names if "present" in name]
    not_kv_inputs = [inp for inp in ov_model.inputs if not any("past_key_values" in n for n in inp.get_names())]

    if not key_value_input_names or not key_value_output_names:
        print("⚠️ No KV cache inputs/outputs found in APM model, skipping stateful transformation")
        return

    batch_dim = 0

    # Add beam_idx for cache reorder support (required by make_stateful)
    input_batch = ov_model.input("input_features").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])

    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(batch_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()

    make_stateful_apm(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
    )


def convert_audio_encoder_stateful(model, output_path: Path):
    """Convert Audio Encoder (MiniCPMWhisperEncoder) to OpenVINO with stateful KV cache.

    This converts the audio encoder with KV cache support and then applies the stateful
    transformation to hide the KV cache inside the model as internal state variables.
    This is analogous to how the LLM language model is converted to stateful mode.

    Benefits over stateless streaming:
    - No need to manually manage KV cache tensors externally
    - Cache is maintained internally by OpenVINO runtime
    - reset_state() to clear cache (equivalent to passing empty cache)
    - Reduced memory copies between host and model
    """
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    apm = model.apm

    def forward_wrap_audio_stateful(
        self,
        input_features,
        attention_mask=None,
        past_key_values=None,
    ):
        """Forward pass with KV cache support (same as streaming version)."""
        # CNN feature extraction
        inputs_embeds = F.gelu(self.conv1(input_features))
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight
        past_key_values_length = 0
        use_cache = True

        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, tuple):
                past_key_values = EncoderDecoderCache(DynamicCache.from_legacy_cache(past_key_values), DynamicCache())
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())

            past_key_values_length = past_key_values.self_attention_cache.get_usable_length(inputs_embeds.shape[1])

            if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:
                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat(
                    (
                        embed_pos_front,
                        torch.repeat_interleave(
                            embed_pos[-1, :].unsqueeze(0),
                            inputs_embeds.shape[1] - embed_pos.shape[0] + past_key_values_length,
                            dim=0,
                        ),
                    )
                )
            else:
                embed_pos = embed_pos[past_key_values_length : inputs_embeds.shape[1] + past_key_values_length, :]
        else:
            embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                layer_head_mask=None,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_encoder_cache = layer_outputs[1]

        hidden_states = self.layer_norm(hidden_states)
        if past_key_values is not None:
            next_encoder_cache = next_encoder_cache.to_legacy_cache()
        return (hidden_states, next_encoder_cache)

    apm._orig_forward = apm.forward
    apm.forward = types.MethodType(forward_wrap_audio_stateful, apm)

    __make_16bit_traceable(apm)

    # Example input for tracing
    batch_size = 1
    mel_length = 300
    mel_bins = 80

    input_features = torch.randn([batch_size, mel_bins, mel_length], dtype=torch.float32)

    seq_len_after_cnn = (mel_length - 1) // 2 + 1

    input_names = ["input_features", "attention_mask"]
    output_names = ["hidden_states"]
    past_key_values = []

    num_layers = apm.config.encoder_layers
    num_heads = apm.config.encoder_attention_heads
    hidden_size = apm.config.d_model
    head_dim = hidden_size // num_heads

    past_seq_len = 2
    for i in range(num_layers):
        key = torch.randn(batch_size, num_heads, past_seq_len, head_dim, dtype=torch.float32)
        value = torch.randn(batch_size, num_heads, past_seq_len, head_dim, dtype=torch.float32)
        past_key_values.append([key, value])
        input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names.extend([f"present.{i}.key", f"present.{i}.value"])

    total_seq_len = past_seq_len + seq_len_after_cnn
    attention_mask = torch.zeros([batch_size, 1, seq_len_after_cnn, total_seq_len], dtype=torch.float32)

    example_input = {
        "input_features": input_features,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
    }

    input_shapes = [
        ov.PartialShape([-1, mel_bins, -1]),
        ov.PartialShape([-1, 1, -1, -1]),
    ]
    input_shapes += [ov.PartialShape([-1, num_heads, -1, head_dim])] * 2 * num_layers

    ov_model = ov.convert_model(
        apm,
        example_input=example_input,
        input=input_shapes,
    )

    for input, input_name in zip(ov_model.inputs, input_names):
        input.get_tensor().set_names({input_name})

    for output, output_name in zip(ov_model.outputs, output_names):
        output.get_tensor().set_names({output_name})

    # Apply stateful transformation: KV cache is hidden inside the model
    patch_stateful_apm(ov_model)
    print("✅ Audio Encoder model converted to stateful mode")

    ov.save_model(ov_model, output_path)

    apm.forward = apm._orig_forward
    del apm._orig_forward
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def convert_audio_projection(model, output_path: Path):
    """Convert Audio Projection Layer to OpenVINO."""
    audio_projection = model.audio_projection_layer

    __make_16bit_traceable(audio_projection)

    # Get dimensions from config
    audio_output_dim = int(model.apm.config.encoder_ffn_dim // 4)

    example_input = torch.randn([1, 100, audio_output_dim], dtype=audio_projection.linear1.weight.dtype)

    # Input shapes: fix audio_output_dim, allow dynamic batch and seq_len
    input_shapes = [
        ov.PartialShape([-1, -1, audio_output_dim]),  # [batch, seq_len, audio_output_dim]
    ]

    ov_model = ov.convert_model(
        audio_projection,
        example_input=example_input,
        input=input_shapes,
    )

    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def convert_tts_text_embedding(model, output_path: Path):
    """Convert TTS Text Embedding to OpenVINO."""
    tts_emb_text = model.tts.emb_text

    __make_16bit_traceable(tts_emb_text)

    ov_model = ov.convert_model(
        tts_emb_text,
        example_input=torch.ones([1, 10], dtype=torch.int64),
    )

    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def convert_tts_language_model(model, output_path: Path, quantization_config=None):
    """Convert TTS Language Model (LlamaModel) to OpenVINO with stateful KV cache."""

    def forward_wrap_tts(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
    ):
        if past_key_values is not None:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        if past_key_values is not None:
            outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()

        hidden_states = outputs.last_hidden_state

        return (hidden_states, outputs.past_key_values)

    tts_model = model.tts
    config = tts_model.config
    hidden_size = config.hidden_size

    patch_cos_sin_cached_fp32(tts_model.model)

    tts_model._orig_forward = tts_model.forward
    tts_model.forward = types.MethodType(forward_wrap_tts, tts_model)

    num_pkv = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    pkv_shape = (2, num_kv_heads, 2, head_dim)

    # Use FP32 for example inputs (will be converted by __make_16bit_traceable)
    inputs_embeds = torch.randn((2, 2, hidden_size), dtype=torch.float32)
    attention_mask = torch.ones([2, 4], dtype=torch.int64)
    position_ids = torch.arange(2, 4).unsqueeze(0).expand(2, -1)

    input_names = ["attention_mask", "position_ids"]
    output_names = ["hidden_states"]
    past_key_values = []

    for i in range(num_pkv):
        kv = [torch.randn(pkv_shape, dtype=torch.float32) for _ in range(2)]
        past_key_values.append(kv)
        input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names.extend([f"present.{i}.key", f"present.{i}.value"])
    input_names.append("inputs_embeds")

    example_input = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    }

    input_shapes = [
        ov.PartialShape([-1, -1]),  # attention_mask
        ov.PartialShape([-1, -1]),  # position_ids
    ]
    input_shapes += [ov.PartialShape([-1, num_kv_heads, -1, head_dim])] * 2 * num_pkv
    input_shapes += [ov.PartialShape([-1, -1, hidden_size])]  # inputs_embeds

    __make_16bit_traceable(tts_model)

    ov_model = ov.convert_model(tts_model, example_input=example_input, input=input_shapes)

    for input, input_name in zip(ov_model.inputs, input_names):
        input.get_tensor().set_names({input_name})

    for output, output_name in zip(ov_model.outputs, output_names):
        output.get_tensor().set_names({output_name})

    # Apply stateful transformation: KV cache is hidden inside the model as internal variables.
    # dim=1 because the first 1 output (hidden_states) is NOT KV cache.
    patch_stateful(ov_model, 1)
    print("✅ TTS Language Model converted to stateful mode")

    if quantization_config is not None and nncf is not None:
        print(f"⌛ TTS Weights compression with {quantization_config.get('mode', 'int8')} mode started")
        ov_model = nncf.compress_weights(ov_model, **_prepare_nncf_config(quantization_config))
        print("✅ TTS Weights compression finished")

    ov.save_model(ov_model, output_path)

    tts_model.forward = tts_model._orig_forward
    del tts_model._orig_forward
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def convert_tts_projector_spk(model, output_path: Path):
    """Convert TTS Speaker Projector to OpenVINO."""
    projector_spk = model.tts.projector_spk

    __make_16bit_traceable(projector_spk)

    # Get LLM dimension for speaker embedding
    llm_dim = model.tts.config.llm_dim

    example_input = torch.randn([1, llm_dim], dtype=projector_spk.linear1.weight.dtype if hasattr(projector_spk, "linear1") else torch.float16)

    # Input shapes: fix llm_dim, allow dynamic batch
    input_shapes = [
        ov.PartialShape([-1, llm_dim]),  # [batch, llm_dim]
    ]

    ov_model = ov.convert_model(
        projector_spk,
        example_input=example_input,
        input=input_shapes,
    )

    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def convert_tts_projector_semantic(model, output_path: Path):
    """Convert TTS Semantic Projector to OpenVINO."""
    projector_semantic = model.tts.projector_semantic

    __make_16bit_traceable(projector_semantic)

    # Get LLM dimension for semantic embedding
    llm_dim = model.tts.config.llm_dim

    example_input = torch.randn([1, 10, llm_dim], dtype=projector_semantic.linear1.weight.dtype if hasattr(projector_semantic, "linear1") else torch.float16)

    # Input shapes: fix llm_dim, allow dynamic batch and seq_len
    input_shapes = [
        ov.PartialShape([-1, -1, llm_dim]),  # [batch, seq_len, llm_dim]
    ]

    ov_model = ov.convert_model(
        projector_semantic,
        example_input=example_input,
        input=input_shapes,
    )

    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def convert_tts_code_embedding(model, output_path: Path):
    """Convert TTS Audio Code Embedding to OpenVINO."""

    class TTSCodeEmbedding(nn.Module):
        def __init__(self, emb_code_list):
            super().__init__()
            self.emb_code = nn.ModuleList(emb_code_list)
            self.num_vq = len(emb_code_list)

        def forward(self, audio_codes):
            """
            audio_codes: [batch, num_vq] or [batch, seq_len, num_vq]
            Returns: summed embeddings
            """
            # Handle both 2D and 3D inputs
            if len(audio_codes.shape) == 2:
                # [batch, num_vq] -> [batch, 1, num_vq]
                audio_codes = audio_codes.unsqueeze(1)

            code_emb = []
            for q in range(self.num_vq):
                x = self.emb_code[q](audio_codes[:, :, q])
                code_emb.append(x)

            # Sum all VQ embeddings
            inputs_embeds = torch.stack(code_emb, dim=-1).sum(-1)
            return inputs_embeds

    tts_code_emb = TTSCodeEmbedding([emb for emb in model.tts.emb_code])

    __make_16bit_traceable(tts_code_emb)

    num_vq = model.tts.num_vq
    example_input = torch.ones([1, 10, num_vq], dtype=torch.int64)

    # Input shapes: fix num_vq, allow dynamic batch and seq_len
    input_shapes = [
        ov.PartialShape([-1, -1, num_vq]),  # audio_codes: [batch, seq_len, num_vq]
    ]

    ov_model = ov.convert_model(
        tts_code_emb,
        example_input=example_input,
        input=input_shapes,
    )

    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def convert_tts_code_head(model, output_path: Path):
    """Convert TTS Audio Code Head to OpenVINO."""

    class TTSCodeHead(nn.Module):
        def __init__(self, head_code_list):
            super().__init__()
            self.head_code = nn.ModuleList(head_code_list)
            self.num_vq = len(head_code_list)

        def forward(self, hidden_states):
            """
            hidden_states: [batch, seq_len, hidden_size]
            Returns: logits [batch, seq_len, num_audio_tokens, num_vq]
            """
            batch_size, seq_len, hidden_size = hidden_states.shape
            num_audio_tokens = self.head_code[0].weight.shape[0]

            logits = torch.empty(
                batch_size,
                seq_len,
                num_audio_tokens,
                self.num_vq,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            for q in range(self.num_vq):
                logits[..., q] = self.head_code[q](hidden_states)

            return logits

    tts_code_head = TTSCodeHead([head for head in model.tts.head_code])

    __make_16bit_traceable(tts_code_head)

    hidden_size = model.tts.config.hidden_size
    num_vq = model.tts.num_vq
    num_audio_tokens = model.tts.config.num_audio_tokens
    # Use FP32 for example input
    example_input = torch.randn([1, 10, hidden_size], dtype=torch.float32)

    # Input shapes: fix hidden_size, allow dynamic batch and seq_len
    # Output will be [batch, seq_len, num_audio_tokens, num_vq]
    input_shapes = [
        ov.PartialShape([-1, -1, hidden_size]),  # hidden_states: [batch, seq_len, hidden_size]
    ]

    ov_model = ov.convert_model(
        tts_code_head,
        example_input=example_input,
        input=input_shapes,
    )

    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


# ==================== Token2wav (Audio Tokenizer) Conversion ====================


def convert_flow_embeddings(token2wav_model, output_path: Path):
    """
    Convert Flow Embeddings model to OpenVINO.

    This includes:
    - Speaker embedding normalization and projection (spk_embed_affine_layer)
    - Token embedding (input_embedding)
    - Encoder (UpsampleConformerEncoderV2)
    - Encoder projection (encoder_proj)

    Args:
        token2wav_model: Token2wav instance containing the flow model
        output_path: Path to save the converted model
    """
    import torch.nn.functional as F

    flow = token2wav_model.flow

    class FlowEmbeddingsWrapper(nn.Module):
        def __init__(self, flow_model):
            super().__init__()
            self.input_embedding = flow_model.input_embedding
            self.spk_embed_affine_layer = flow_model.spk_embed_affine_layer
            self.encoder = flow_model.encoder
            self.encoder_proj = flow_model.encoder_proj
            self.up_rate = flow_model.up_rate

        def forward(
            self,
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            embedding,
        ):
            """
            Args:
                token: speech tokens [batch, seq_len]
                token_len: token lengths [batch]
                prompt_token: prompt speech tokens [batch, prompt_len]
                prompt_token_len: prompt token lengths [batch]
                embedding: speaker embedding [batch, 192]
            Returns:
                h: encoder output [batch, mel_len, output_size]
                spks: projected speaker embedding [batch, output_size]
            """
            # Normalize and project speaker embedding
            embedding = F.normalize(embedding, dim=1)
            spks = self.spk_embed_affine_layer(embedding)  # (batch, output_size)

            # Concatenate prompt_token and token
            token = torch.concat([prompt_token, token], dim=1)
            token_len = prompt_token_len + token_len

            # Create mask and embed tokens
            # Simple mask creation without make_pad_mask
            batch_size = token.shape[0]
            max_len = token.shape[1]
            seq_range = torch.arange(0, max_len, device=token.device).unsqueeze(0).expand(batch_size, -1)
            mask = (seq_range < token_len.unsqueeze(-1)).unsqueeze(-1).to(spks.dtype)

            token = self.input_embedding(torch.clamp(token, min=0)) * mask

            # Encode tokens
            h, _ = self.encoder.forward(token, token_len)
            h = self.encoder_proj(h)

            return h, spks

    flow_emb_wrapper = FlowEmbeddingsWrapper(flow)
    flow_emb_wrapper.eval()

    __make_16bit_traceable(flow_emb_wrapper)

    # Get output_size from flow model
    output_size = flow.output_size if hasattr(flow, "output_size") else 80

    # Example inputs
    example_input = {
        "token": torch.ones([1, 50], dtype=torch.int32),
        "token_len": torch.tensor([50], dtype=torch.int32),
        "prompt_token": torch.ones([1, 30], dtype=torch.int32),
        "prompt_token_len": torch.tensor([30], dtype=torch.int32),
        "embedding": torch.randn([1, 192], dtype=torch.float32),
    }

    # Input shapes: fix embedding dim=192, allow dynamic batch and seq_len
    input_shapes = [
        ov.PartialShape([-1, -1]),  # token: [batch, seq_len]
        ov.PartialShape([-1]),  # token_len: [batch]
        ov.PartialShape([-1, -1]),  # prompt_token: [batch, prompt_len]
        ov.PartialShape([-1]),  # prompt_token_len: [batch]
        ov.PartialShape([-1, 192]),  # embedding: [batch, 192]
    ]

    with torch.no_grad():
        ov_model = ov.convert_model(flow_emb_wrapper, example_input=example_input, input=input_shapes)

    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"✅ Flow Embeddings model saved to {output_path}")


def convert_flow_estimator(token2wav_model, output_path: Path):
    """
    Convert Flow Estimator (DiT decoder) model to OpenVINO.

    This is the diffusion transformer that performs flow matching denoising.

    Args:
        token2wav_model: Token2wav instance containing the flow model
        output_path: Path to save the converted model
    """
    flow = token2wav_model.flow
    estimator = flow.decoder.estimator

    # Patch F.scaled_dot_product_attention to convert bool masks to float masks
    import torch.nn.functional as F

    original_sdpa = F.scaled_dot_product_attention

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        """Patched SDPA that converts bool masks to float masks for OpenVINO compatibility."""
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            # Convert boolean mask to float mask with -inf for masked positions
            attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill(~attn_mask, float("-inf"))
        return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    # Temporarily replace F.scaled_dot_product_attention
    F.scaled_dot_product_attention = patched_sdpa

    # Get output_size from flow model
    output_size = flow.output_size if hasattr(flow, "output_size") else 80

    # Example inputs for estimator
    # x: [batch*2, output_size, mel_len] - doubled for CFG
    # mask: [batch*2, 1, mel_len]
    # mu: [batch*2, output_size, mel_len]
    # t: [batch*2]
    # spks: [batch*2, output_size]
    # cond: [batch*2, output_size, mel_len]
    example_input = {
        "x": torch.randn([2, output_size, 200], dtype=torch.float32),
        "mask": torch.ones([2, 1, 200], dtype=torch.float32),
        "mu": torch.randn([2, output_size, 200], dtype=torch.float32),
        "t": torch.tensor([0.5, 0.5], dtype=torch.float32),
        "spks": torch.randn([2, output_size], dtype=torch.float32),
        "cond": torch.randn([2, output_size, 200], dtype=torch.float32),
    }

    # Input shapes: fix output_size, allow dynamic mel_len
    input_shapes = [
        ov.PartialShape([2, output_size, -1]),  # x: [2, output_size, mel_len]
        ov.PartialShape([2, 1, -1]),  # mask: [2, 1, mel_len]
        ov.PartialShape([2, output_size, -1]),  # mu: [2, output_size, mel_len]
        ov.PartialShape([2]),  # t: [2]
        ov.PartialShape([2, output_size]),  # spks: [2, output_size]
        ov.PartialShape([2, output_size, -1]),  # cond: [2, output_size, mel_len]
    ]

    estimator.eval()
    __make_16bit_traceable(estimator)

    with torch.no_grad():
        ov_model = ov.convert_model(estimator, example_input=example_input, input=input_shapes)

    # Restore original F.scaled_dot_product_attention
    F.scaled_dot_product_attention = original_sdpa

    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"✅ Flow Estimator model saved to {output_path}")


def convert_hift(token2wav_model, output_path: Path):
    """
    Convert HiFT vocoder model to OpenVINO.

    The HiFT model converts mel spectrograms to waveforms.
    We export the neural network part (before istft) to avoid complex ops.

    Args:
        token2wav_model: Token2wav instance containing the hift model
        output_path: Path to save the converted model
    """
    import numpy as np

    hift = token2wav_model.hift

    class HiFTWrapper(nn.Module):
        """
        HiFT forward wrapper that outputs raw spectral features before istft.
        This allows istft to be performed in Python with better compatibility.
        """

        def __init__(self, hift_model):
            super().__init__()
            self.f0_predictor = hift_model.f0_predictor
            self.f0_upsamp = hift_model.f0_upsamp
            self.m_source = hift_model.m_source
            self.conv_pre = hift_model.conv_pre
            self.ups = hift_model.ups
            self.source_downs = hift_model.source_downs
            self.source_resblocks = hift_model.source_resblocks
            self.resblocks = hift_model.resblocks
            self.conv_post = hift_model.conv_post
            self.reflection_pad = hift_model.reflection_pad
            self.num_upsamples = hift_model.num_upsamples
            self.num_kernels = hift_model.num_kernels
            self.lrelu_slope = hift_model.lrelu_slope
            self.istft_params = hift_model.istft_params
            self.stft_window = hift_model.stft_window

        def _stft(self, x):
            spec = torch.stft(
                x,
                self.istft_params["n_fft"],
                self.istft_params["hop_len"],
                self.istft_params["n_fft"],
                window=self.stft_window.to(x.device),
                return_complex=True,
            )
            spec = torch.view_as_real(spec)  # [B, F, TT, 2]
            return spec[..., 0], spec[..., 1]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            HiFT forward that outputs raw conv_post output before istft.

            Input: mel spectrogram (batch, 80, mel_len)
            Output: raw spectral features (batch, n_fft+2, time) before istft
            """
            # mel -> f0
            f0 = self.f0_predictor(x)
            # f0 -> source
            s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
            s, _, _ = self.m_source(s)
            s = s.transpose(1, 2)

            # stft of source
            s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
            s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

            # Run decode up to conv_post only
            x = self.conv_pre(x)
            for i in range(self.num_upsamples):
                x = F.leaky_relu(x, self.lrelu_slope)
                x = self.ups[i](x)

                if i == self.num_upsamples - 1:
                    x = self.reflection_pad(x)

                # fusion with source
                si = self.source_downs[i](s_stft)
                si = self.source_resblocks[i](si)
                x = x + si

                xs = None
                for j in range(self.num_kernels):
                    if xs is None:
                        xs = self.resblocks[i * self.num_kernels + j](x)
                    else:
                        xs += self.resblocks[i * self.num_kernels + j](x)
                x = xs / self.num_kernels

            x = F.leaky_relu(x)
            x = self.conv_post(x)
            # Return here without exp, sin, istft, clamp
            return x

    hift_wrapper = HiFTWrapper(hift)
    hift_wrapper.eval()

    __make_16bit_traceable(hift_wrapper)

    # Example input: mel spectrogram [batch, 80, mel_len]
    mel_bins = 80
    example_input = torch.randn([1, mel_bins, 200], dtype=torch.float32)

    # Input shapes: fix mel_bins=80, allow dynamic batch and mel_len
    input_shapes = [
        ov.PartialShape([-1, mel_bins, -1]),  # [batch, mel_bins=80, mel_len]
    ]

    with torch.no_grad():
        ov_model = ov.convert_model(hift_wrapper, example_input=example_input, input=input_shapes)

    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"✅ HiFT model saved to {output_path}")


def convert_token2wav(
    audio_tokenizer_path: str,
    output_dir: str,
    float16: bool = False,
):
    """
    Convert Token2wav (audio_tokenizer) to OpenVINO format.

    This converts the following models:
    1. Flow Embeddings (input_embedding + encoder + spk_embed_affine_layer)
    2. Flow Estimator (DiT decoder for flow matching)
    3. HiFT (vocoder for mel to waveform conversion)

    Note: s3tokenizer and campplus are already ONNX models and can be used directly.

    Args:
        audio_tokenizer_path: Path to audio_tokenizer model directory
        output_dir: Directory to save converted models
        float16: Whether to use float16 for the PyTorch models

    Returns:
        Path to output directory
    """
    import sys
    import types

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    flow_emb_path = output_path / FLOW_EMBEDDINGS_NAME
    flow_est_path = output_path / FLOW_ESTIMATOR_NAME
    hift_path = output_path / HIFT_NAME

    # Check if all models exist
    if flow_emb_path.exists() and flow_est_path.exists() and hift_path.exists():
        print(f"✅ Token2wav models already converted. You can find results in {output_dir}")
        return output_path

    print(f"⌛ Token2wav conversion started...")

    # Setup cosyvoice2 alias for hyperpyyaml loading
    def _setup_cosyvoice2_alias():
        if "cosyvoice2.flow.flow" in sys.modules:
            return

        import stepaudio2.cosyvoice2.flow.flow as _step_flow
        import stepaudio2.cosyvoice2.flow.flow_matching as _step_flow_matching
        import stepaudio2.cosyvoice2.flow.decoder_dit as _step_decoder_dit
        import stepaudio2.cosyvoice2.transformer.upsample_encoder_v2 as _step_upsample

        cosyvoice2_pkg = types.ModuleType("cosyvoice2")
        cosyvoice2_flow_pkg = types.ModuleType("cosyvoice2.flow")
        cosyvoice2_transformer_pkg = types.ModuleType("cosyvoice2.transformer")

        cosyvoice2_flow_pkg.flow = _step_flow
        cosyvoice2_flow_pkg.flow_matching = _step_flow_matching
        cosyvoice2_flow_pkg.decoder_dit = _step_decoder_dit
        cosyvoice2_transformer_pkg.upsample_encoder_v2 = _step_upsample

        cosyvoice2_pkg.flow = cosyvoice2_flow_pkg
        cosyvoice2_pkg.transformer = cosyvoice2_transformer_pkg

        sys.modules["cosyvoice2"] = cosyvoice2_pkg
        sys.modules["cosyvoice2.flow"] = cosyvoice2_flow_pkg
        sys.modules["cosyvoice2.flow.flow"] = _step_flow
        sys.modules["cosyvoice2.flow.flow_matching"] = _step_flow_matching
        sys.modules["cosyvoice2.flow.decoder_dit"] = _step_decoder_dit
        sys.modules["cosyvoice2.transformer"] = cosyvoice2_transformer_pkg
        sys.modules["cosyvoice2.transformer.upsample_encoder_v2"] = _step_upsample

    _setup_cosyvoice2_alias()

    # Load Token2wav model
    from hyperpyyaml import load_hyperpyyaml
    from stepaudio2.flashcosyvoice.modules.hifigan import HiFTGenerator

    print("⌛ Loading Token2wav model...")

    # Load flow model
    with open(f"{audio_tokenizer_path}/flow.yaml", "r") as f:
        configs = load_hyperpyyaml(f)
        flow = configs["flow"]

    if float16:
        flow.half()
    flow.load_state_dict(torch.load(f"{audio_tokenizer_path}/flow.pt", map_location="cpu", weights_only=True), strict=True)
    flow.cpu().eval()

    # Load HiFT model
    hift = HiFTGenerator()
    hift_state_dict = {k.replace("generator.", ""): v for k, v in torch.load(f"{audio_tokenizer_path}/hift.pt", map_location="cpu", weights_only=True).items()}
    hift.load_state_dict(hift_state_dict, strict=True)
    hift.cpu().eval()

    # Create a simple namespace to hold the models
    class Token2wavWrapper:
        def __init__(self, flow, hift):
            self.flow = flow
            self.hift = hift

    token2wav = Token2wavWrapper(flow, hift)
    print("✅ Token2wav model loaded")

    # Convert Flow Embeddings
    if not flow_emb_path.exists():
        print("⌛ Converting Flow Embeddings model...")
        convert_flow_embeddings(token2wav, flow_emb_path)
    else:
        print(f"✅ Flow Embeddings model already exists at {flow_emb_path}")

    # Convert Flow Estimator
    if not flow_est_path.exists():
        print("⌛ Converting Flow Estimator model...")
        convert_flow_estimator(token2wav, flow_est_path)
    else:
        print(f"✅ Flow Estimator model already exists at {flow_est_path}")

    # Convert HiFT
    if not hift_path.exists():
        print("⌛ Converting HiFT model...")
        convert_hift(token2wav, hift_path)
    else:
        print(f"✅ HiFT model already exists at {hift_path}")

    # Copy ONNX models (s3tokenizer and campplus)
    # These models are small and kept as ONNX for compatibility with s3tokenizer library
    import shutil

    onnx_files = ["speech_tokenizer_v2_25hz.onnx", "campplus.onnx"]
    for onnx_file in onnx_files:
        src = Path(audio_tokenizer_path) / onnx_file
        dst = output_path / onnx_file
        if src.exists():
            if not dst.exists():
                print(f"⌛ Copying {onnx_file}...")
                shutil.copy2(src, dst)
                print(f"✅ {onnx_file} copied to {dst}")
            else:
                print(f"✅ {onnx_file} already exists at {dst}")
        else:
            print(f"⚠️ Warning: {onnx_file} not found at {src}")
            print(f"   Model may not work properly without this file.")

    del token2wav, flow, hift
    gc.collect()

    print(f"✅ Token2wav conversion finished. You can find results in {output_dir}")
    return output_path


# ==================== Utility functions for inference ====================


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=-1)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    out = np.einsum("hw,d->hwd", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)
    return emb


def torch_clone_recursive(data, dtype=None):
    """Recursively clone a nested list/tuple structure with tensors."""
    if isinstance(data, torch.Tensor):
        result = data.clone()
        if dtype is not None:
            result = result.to(dtype)
        return result
    elif isinstance(data, list):
        return [torch_clone_recursive(item, dtype=dtype) for item in data]
    elif isinstance(data, tuple):
        return tuple(torch_clone_recursive(item, dtype=dtype) for item in data)
    else:
        return data


# ==================== OpenVINO Model Wrappers ====================


class OVLLMEmbedding:
    """OpenVINO wrapper for LLM embedding model."""

    def __init__(self, model_path, device):
        self.model = Core().compile_model(model_path, device)
        self.request = self.model.create_infer_request()
        # Get actual input name (typically 'input')
        self._input_name = self.model.inputs[0].get_any_name()

    def __call__(self, input_ids):
        """Get token embeddings. Returns torch tensor."""
        # OpenVINO directly accepts torch.Tensor, no conversion needed
        data = input_ids

        # Ensure input is 2D (batch_size, seq_len)
        # If 1D, reshape to (1, seq_len)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self.request.infer({self._input_name: data})

        output = torch.from_numpy(self.request.get_output_tensor(0).data.copy())

        # If input was originally 1D, squeeze the batch dimension
        if output.shape[0] == 1 and len(input_ids.shape if isinstance(input_ids, torch.Tensor) else input_ids.shape) == 1:
            output = output.squeeze(0)

        return output


class OVLLMLanguageModel:
    """OpenVINO wrapper for LLM language model.

    Auto-detects stateful vs stateless models:
    - Stateful: KV cache is managed internally by OpenVINO as state variables.
      Only inputs_embeds/attention_mask/position_ids/beam_idx are needed.
      reset_state() calls request.reset_state() to clear internal cache.
    - Stateless (fallback): KV cache passed/received explicitly as model I/O.
      Used when sliding window needs to manipulate cache directly.
    """

    def __init__(self, model_path, device):
        self.model = Core().compile_model(model_path, device)
        self.request = self.model.create_infer_request()
        self.input_names = {input_t.get_any_name() for input_t in self.model.inputs}
        self.output_names = {output_t.get_any_name() for output_t in self.model.outputs}

        # Auto-detect stateful model: stateful models have no past_key_values inputs
        # (KV cache is hidden as internal state variables)
        self.key_value_input_names = [name for name in self.input_names if "past_key_values" in name]
        self._is_stateful = len(self.key_value_input_names) == 0

        if self._is_stateful:
            self.num_layers = 0
            self.past_key_values = None  # Not used, kept for API compat
            print("✅ LLM Language Model: stateful mode (KV cache managed internally)")
        else:
            self.num_layers = sum(1 for name in self.output_names if ".key" in name)
            self.past_key_values = None
            print(f"✅ LLM Language Model: stateless mode ({self.num_layers} KV layers)")

    def reset_state(self):
        """Reset KV cache state."""
        if self._is_stateful:
            self.request.reset_state()
        else:
            self.past_key_values = None

    def __call__(self, inputs_embeds, attention_mask, position_ids):
        """Run LLM forward pass.

        Returns:
            tuple: (logits, hidden_states) where:
                - logits: [batch, seq_len, vocab_size]
                - hidden_states: [batch, seq_len, hidden_size] (last layer output)
        """
        if self._is_stateful:
            return self._call_stateful(inputs_embeds, attention_mask, position_ids)
        else:
            return self._call_stateless(inputs_embeds, attention_mask, position_ids)

    def _call_stateful(self, inputs_embeds, attention_mask, position_ids):
        """Stateful inference: only pass data inputs, KV cache is internal."""
        inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        # beam_idx required for stateful models (even for greedy, batch=1)
        if "beam_idx" in self.input_names:
            batch_size = inputs_embeds.shape[0]
            inputs["beam_idx"] = np.arange(batch_size, dtype=np.int32)

        self.request.infer(inputs)

        logits = torch.from_numpy(self.request.get_output_tensor(0).data.copy())
        hidden_states = torch.from_numpy(self.request.get_output_tensor(1).data.copy())

        return logits, hidden_states

    def _call_stateless(self, inputs_embeds, attention_mask, position_ids):
        """Stateless inference: manually pass/receive KV cache."""
        inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        if self.past_key_values is not None:
            for layer_idx in range(self.num_layers):
                key, value = self.past_key_values[layer_idx]
                inputs[f"past_key_values.{layer_idx}.key"] = key
                inputs[f"past_key_values.{layer_idx}.value"] = value
        else:
            shape_input_ids = inputs_embeds.shape
            batch_size = shape_input_ids[0]
            num_attention_heads = 1

            for input_name in self.key_value_input_names:
                model_input = self.model.input(input_name)
                shape = model_input.get_partial_shape()
                shape[0] = batch_size * num_attention_heads
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[1].is_dynamic:
                    shape[1] = 0
                inputs[input_name] = ov.Tensor(model_input.get_element_type(), shape.get_shape())

        self.request.infer(inputs)

        logits = torch.from_numpy(self.request.get_output_tensor(0).data.copy())
        hidden_states = torch.from_numpy(self.request.get_output_tensor(1).data.copy())

        # ZERO-COPY: get views of OV's output buffers for KV cache
        present_key_values = []
        for layer_idx in range(self.num_layers):
            key = np.array(self.request.get_tensor(f"present.{layer_idx}.key").data, copy=False)
            value = np.array(self.request.get_tensor(f"present.{layer_idx}.value").data, copy=False)
            present_key_values.append((key, value))

        self.past_key_values = present_key_values

        return logits, hidden_states


class OVVisionModel:
    """OpenVINO wrapper for vision encoder.

    The VPM model accepts pre-computed position_ids instead of tgt_sizes,
    because the original SiglipVisionEmbeddings uses boolean-mask indexing
    that doesn't trace correctly for dynamic shapes.
    """

    def __init__(self, model_path, device):
        self.model = Core().compile_model(model_path, device)
        self.request = self.model.create_infer_request()

    def __call__(self, pixel_values, patch_attention_mask, position_ids):
        """Encode images. Returns torch tensor.

        Args:
            pixel_values: [B, 3, H, W] image patches
            patch_attention_mask: [B, 1, num_patches] boolean mask
            position_ids: [B, num_patches] pre-computed position IDs
        """
        inputs = {
            "pixel_values": pixel_values,
            "patch_attention_mask": patch_attention_mask,
            "position_ids": position_ids,
        }
        start_time = time.perf_counter()
        self.request.infer(inputs)
        infer_time = (time.perf_counter() - start_time) * 1000
        return torch.from_numpy(self.request.get_output_tensor(0).data.copy())


class OVResampler:
    """OpenVINO wrapper for resampler.

    The exported OV model takes (x, pos_embed) where:
    - x: [1, num_patches, vision_dim]
    - pos_embed: [num_patches, 1, embed_dim]

    Position embeddings are precomputed here (matching original model).
    Images are processed one at a time since the exported model uses batch=1.
    """

    def __init__(self, model_path, device, embed_dim=4096, max_size=(70, 70)):
        self.model = Core().compile_model(model_path, device)
        self.request = self.model.create_infer_request()

        # Pre-compute position embeddings (same as original model)
        self.max_size = max_size
        self.pos_embed_base = get_2d_sincos_pos_embed(embed_dim, self.max_size)
        self.pos_embed = torch.from_numpy(self.pos_embed_base).float()

    def __call__(self, x, tgt_sizes):
        """Resample visual features.

        Args:
            x: visual features [B, L, vision_dim] from vision encoder
            tgt_sizes: target sizes [B, 2] (height, width per image)

        Returns:
            Resampled features [B, num_queries, embed_dim]
        """
        bs = x.shape[0]

        results = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            num_patches = tgt_h.item() * tgt_w.item()

            # Get position embeddings for this image's patch grid
            pos_embed_2d = self.pos_embed[: tgt_h.item(), : tgt_w.item(), :].reshape(num_patches, -1)

            # Get visual features for this image (only valid patches)
            xi = x[i : i + 1, :num_patches, :]  # [1, num_patches, vision_dim]

            # pos_embed shape: [num_patches, 1, embed_dim] (as expected by exported model)
            pos_embed_input = pos_embed_2d.unsqueeze(1)  # [num_patches, 1, embed_dim]

            # OpenVINO directly accepts torch.Tensor
            inputs = {
                "x": xi,
                "pos_embed": pos_embed_input,
            }
            self.request.infer(inputs)
            result = torch.from_numpy(self.request.get_output_tensor(0).data.copy())
            results.append(result)

        return torch.cat(results, dim=0)  # [B, num_queries, embed_dim]


class OVAudioEncoder:
    """OpenVINO wrapper for audio encoder with KV cache support.

    Supports three modes (auto-detected from model structure):
    1. Standard mode: No KV cache inputs/outputs (stateless, no streaming)
    2. Streaming mode: Explicit KV cache inputs/outputs (stateless with manual cache mgmt)
    3. Stateful mode: KV cache hidden as internal state variables (like stateful LLM)
       - reset_state() clears internal cache
       - No need to pass/receive KV cache tensors
       - Tracks cached sequence length via _cached_seq_len counter
    """

    def __init__(self, model_path, device):
        self.model = Core().compile_model(model_path, device)
        self.request = self.model.create_infer_request()

        # Detect model mode by checking inputs
        input_names = [inp.get_any_name() for inp in self.model.inputs]
        self.supports_kv_cache = any("past_key_values" in name for name in input_names)
        self.key_value_input_names = [name for name in input_names if "past_key_values" in name]

        # Detect stateful mode: has beam_idx but no past_key_values inputs
        # (stateful transformation removes past_key_values inputs and hides them as state)
        self._has_beam_idx = "beam_idx" in input_names
        self._is_stateful = self._has_beam_idx and not self.supports_kv_cache

        # Track cached sequence length for stateful mode attention mask construction
        self._cached_seq_len = 0

        if self._is_stateful:
            self.num_layers = 0  # Not needed, cache is internal
            print("✅ Audio encoder loaded in stateful mode (KV cache managed internally)")
        elif self.supports_kv_cache:
            # Parse cache configuration from model inputs
            self.num_layers = sum(1 for name in input_names if ".key" in name)
            self.cache_input_names = [name for name in input_names if "past_key_values" in name]
            self.cache_output_names = [out.get_any_name() for out in self.model.outputs if "present" in out.get_any_name()]
            print(f"✅ Audio encoder loaded with KV cache support ({self.num_layers} layers)")
        else:
            print("✅ Audio encoder loaded in standard mode (no KV cache)")

    def reset_state(self):
        """Reset internal KV cache state (stateful mode) or clear cached values."""
        if self._is_stateful:
            self.request.reset_state()
        self._cached_seq_len = 0

    def get_cached_seq_len(self):
        """Get the current cached sequence length (for stateful mode)."""
        return self._cached_seq_len

    def __call__(self, input_features, attention_mask=None, past_key_values=None, use_cache=True):
        """Encode audio features with optional KV cache.

        Args:
            input_features: mel spectrogram [batch, 80, mel_length]
            attention_mask: attention mask [batch, 1, seq_len, total_seq_len] (for streaming)
            past_key_values: list of [key, value] pairs for each layer (legacy cache format)
                             Ignored in stateful mode.
            use_cache: whether to return updated cache

        Returns:
            If use_cache=False: hidden_states tensor
            If use_cache=True and stateful: hidden_states tensor (cache is internal)
            If use_cache=True and stateless: (hidden_states, updated_past_key_values)
        """
        if self._is_stateful:
            return self._call_stateful(input_features, attention_mask, use_cache)
        elif self.supports_kv_cache:
            return self._call_streaming(input_features, attention_mask, past_key_values, use_cache)
        else:
            return self._call_standard(input_features, attention_mask)

    def _call_stateful(self, input_features, attention_mask, use_cache):
        """Stateful inference: KV cache is managed internally by OpenVINO.

        For use_cache=False (non-streaming), we reset_state() before AND after inference
        to ensure the internal state doesn't leak into subsequent streaming calls.
        For use_cache=True (streaming), the internal state accumulates across calls.
        """
        if not use_cache:
            # Non-streaming: reset state for independent encoding
            self.reset_state()

        inputs = {"input_features": input_features}

        # Construct attention mask if not provided
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        else:
            # Build a zero attention mask (no masking) for the full sequence
            # After CNN: seq_len = (mel_length - 1) // 2 + 1
            mel_length = input_features.shape[2]
            current_seq_len = (mel_length - 1) // 2 + 1
            total_seq_len = self._cached_seq_len + current_seq_len
            batch_size = input_features.shape[0]
            inputs["attention_mask"] = torch.zeros(
                (batch_size, 1, current_seq_len, total_seq_len),
                dtype=torch.float32,
            )

        # beam_idx required for stateful models
        if self._has_beam_idx:
            batch_size = input_features.shape[0]
            inputs["beam_idx"] = np.arange(batch_size, dtype=np.int32)

        start_time = time.perf_counter()
        self.request.infer(inputs)
        infer_time = (time.perf_counter() - start_time) * 1000

        hidden_states = torch.from_numpy(self.request.get_output_tensor(0).data.copy())

        # Update cached sequence length tracker
        if use_cache:
            # After CNN: seq_len_after_cnn = (mel_length - 1) // 2 + 1
            # hidden_states shape: [batch, seq_len_after_cnn, hidden_size]
            self._cached_seq_len += hidden_states.shape[1]
        else:
            # Non-streaming: reset state after inference to prevent internal KV cache
            # from leaking into subsequent streaming calls
            self.request.reset_state()
            self._cached_seq_len = 0

        return hidden_states

    def _call_streaming(self, input_features, attention_mask, past_key_values, use_cache):
        """Streaming (stateless) inference with explicit KV cache management."""
        inputs = {"input_features": input_features}

        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask

        if past_key_values is not None:
            for i, (key, value) in enumerate(past_key_values):
                inputs[f"past_key_values.{i}.key"] = key
                inputs[f"past_key_values.{i}.value"] = value
        else:
            batch_size = input_features.shape[0]
            for input_name in self.key_value_input_names:
                model_input = self.model.input(input_name)
                shape = model_input.get_partial_shape()
                shape[0] = batch_size
                shape[2] = 0
                inputs[input_name] = ov.Tensor(model_input.get_element_type(), shape.get_shape())

        start_time = time.perf_counter()
        self.request.infer(inputs)
        infer_time = (time.perf_counter() - start_time) * 1000

        hidden_states = torch.from_numpy(self.request.get_output_tensor(0).data.copy())

        if use_cache:
            updated_cache = []
            for i in range(self.num_layers):
                key = torch.from_numpy(self.request.get_tensor(f"present.{i}.key").data.copy())
                value = torch.from_numpy(self.request.get_tensor(f"present.{i}.value").data.copy())
                updated_cache.append([key, value])
            return hidden_states, updated_cache
        else:
            return hidden_states

    def _call_standard(self, input_features, attention_mask):
        """Standard (no KV cache) inference."""
        inputs = {"input_features": input_features}
        if attention_mask is not None and "attention_mask" in [inp.get_any_name() for inp in self.model.inputs]:
            inputs["attention_mask"] = attention_mask

        start_time = time.perf_counter()
        self.request.infer(inputs)
        infer_time = (time.perf_counter() - start_time) * 1000
        return torch.from_numpy(self.request.get_output_tensor(0).data.copy())


class OVAudioProjection:
    """OpenVINO wrapper for audio projection layer."""

    def __init__(self, model_path, device):
        self.model = Core().compile_model(model_path, device)
        self.request = self.model.create_infer_request()

    def __call__(self, audio_features):
        """Project audio features. Returns torch tensor."""
        # OpenVINO directly accepts torch.Tensor
        inputs = {"audio_features": audio_features}

        self.request.infer(inputs)

        return torch.from_numpy(self.request.get_output_tensor(0).data.copy())


class OVTTSEmbedding:
    """OpenVINO wrapper for TTS embedding model."""

    def __init__(self, model_path, device):
        self.model = Core().compile_model(model_path, device)
        self.request = self.model.create_infer_request()
        # Get actual input name (typically 'input')
        self._input_name = self.model.inputs[0].get_any_name()

    def __call__(self, input_ids):
        """Get TTS token embeddings. Returns torch tensor.

        Args:
            input_ids: [seq_len] or [batch, seq_len] int64 tensor
        """
        # OpenVINO directly accepts torch.Tensor
        data = input_ids
        # Ensure 2D input [batch, seq_len]
        if data.ndim == 1:
            data = data.reshape(1, -1)
        inputs = {self._input_name: data}
        self.request.infer(inputs)
        result = torch.from_numpy(self.request.get_output_tensor(0).data.copy())
        # If input was 1D, squeeze batch dimension from output
        if input_ids.dim() == 1:
            result = result.squeeze(0)
        return result


class OVTTSLanguageModel:
    """OpenVINO wrapper for TTS language model.

    Auto-detects stateful vs stateless models:
    - Stateful: KV cache is managed internally by OpenVINO. Only data inputs needed.
    - Stateless: KV cache passed/received explicitly (legacy fallback).

    TTS model is a perfect fit for stateful because:
    - Pure autoregressive generation (no cache slicing/sliding window)
    - Cache only needs reset between TTS generation sessions
    """

    def __init__(self, model_path, device):
        self.model = Core().compile_model(model_path, device)
        self.request = self.model.create_infer_request()
        self.input_names = {input_t.get_any_name() for input_t in self.model.inputs}
        self.output_names = {output_t.get_any_name() for output_t in self.model.outputs}

        # Auto-detect stateful model
        self.key_value_input_names = [name for name in self.input_names if "past_key_values" in name]
        self._is_stateful = len(self.key_value_input_names) == 0

        if self._is_stateful:
            self.num_layers = 0
            self.past_key_values = None
            print("✅ TTS Language Model: stateful mode (KV cache managed internally)")
        else:
            self.num_layers = sum(1 for name in self.output_names if ".key" in name)
            self.past_key_values = None
            print(f"✅ TTS Language Model: stateless mode ({self.num_layers} KV layers)")

    def reset_state(self):
        """Reset KV cache state."""
        if self._is_stateful:
            self.request.reset_state()
        else:
            self.past_key_values = None

    def __call__(self, inputs_embeds, attention_mask, position_ids):
        """Run TTS LLM forward pass."""
        if self._is_stateful:
            return self._call_stateful(inputs_embeds, attention_mask, position_ids)
        else:
            return self._call_stateless(inputs_embeds, attention_mask, position_ids)

    def _call_stateful(self, inputs_embeds, attention_mask, position_ids):
        """Stateful inference: only pass data inputs, KV cache is internal."""
        inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        if "beam_idx" in self.input_names:
            batch_size = inputs_embeds.shape[0]
            inputs["beam_idx"] = np.arange(batch_size, dtype=np.int32)

        self.request.infer(inputs)

        hidden_states = torch.from_numpy(self.request.get_output_tensor(0).data.copy())
        return hidden_states

    def _call_stateless(self, inputs_embeds, attention_mask, position_ids):
        """Stateless inference: manually pass/receive KV cache."""
        inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        if self.past_key_values is not None:
            for layer_idx in range(self.num_layers):
                key, value = self.past_key_values[layer_idx]
                inputs[f"past_key_values.{layer_idx}.key"] = key
                inputs[f"past_key_values.{layer_idx}.value"] = value
        else:
            shape_input_ids = inputs_embeds.shape
            batch_size = shape_input_ids[0]
            num_attention_heads = 1

            for input_name in self.key_value_input_names:
                model_input = self.model.input(input_name)
                shape = model_input.get_partial_shape()
                shape[0] = batch_size * num_attention_heads
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[1].is_dynamic:
                    shape[1] = 0
                inputs[input_name] = ov.Tensor(model_input.get_element_type(), shape.get_shape())

        self.request.infer(inputs)

        hidden_states = torch.from_numpy(self.request.get_output_tensor(0).data.copy())

        present_key_values = []
        for layer_idx in range(self.num_layers):
            key = np.array(self.request.get_tensor(f"present.{layer_idx}.key").data, copy=False)
            value = np.array(self.request.get_tensor(f"present.{layer_idx}.value").data, copy=False)
            present_key_values.append((key, value))

        self.past_key_values = present_key_values

        return hidden_states


class OVTTSProjectorSpk:
    """OpenVINO wrapper for TTS speaker projector."""

    def __init__(self, model_path, device):
        self.model = Core().compile_model(model_path, device)
        self.request = self.model.create_infer_request()
        # Get actual input name (typically 'audio_features')
        self._input_name = self.model.inputs[0].get_any_name()

    def __call__(self, spk_emb):
        """Project speaker embedding. Returns torch tensor.

        Args:
            spk_emb: [4096] or [batch, 4096] float tensor
        """
        # OpenVINO directly accepts torch.Tensor
        data = spk_emb
        # Ensure 2D input [batch, hidden_size]
        if data.ndim == 1:
            data = data.reshape(1, -1)
        inputs = {self._input_name: data}
        self.request.infer(inputs)
        return torch.from_numpy(self.request.get_output_tensor(0).data.copy())


class OVTTSProjectorSemantic:
    """OpenVINO wrapper for TTS semantic projector."""

    def __init__(self, model_path, device):
        self.model = Core().compile_model(model_path, device)
        self.request = self.model.create_infer_request()
        # Get actual input name (typically 'audio_features')
        self._input_name = self.model.inputs[0].get_any_name()

    def __call__(self, semantic_emb):
        """Project semantic embedding. Returns torch tensor.

        Args:
            semantic_emb: [seq_len, 4096] or [batch, seq_len, 4096] float tensor
        """
        # OpenVINO directly accepts torch.Tensor
        data = semantic_emb
        # Ensure 3D input [batch, seq_len, hidden_size]
        if data.ndim == 2:
            data = data.reshape(1, *data.shape)
        inputs = {self._input_name: data}
        self.request.infer(inputs)
        result = torch.from_numpy(self.request.get_output_tensor(0).data.copy())
        # If input was 2D, squeeze batch dimension
        if semantic_emb.dim() == 2:
            result = result.squeeze(0)
        return result


class OVTTSCodeEmbedding:
    """OpenVINO wrapper for TTS code embedding."""

    def __init__(self, model_path, device):
        self.model = Core().compile_model(model_path, device)
        self.request = self.model.create_infer_request()
        # Get actual input name (may be auto-generated like '13')
        self._input_name = self.model.inputs[0].get_any_name()

    def __call__(self, codes):
        """Get TTS code embeddings. Returns torch tensor.

        Args:
            codes: [batch, seq_len, num_vq] int64 tensor
        """
        # OpenVINO directly accepts torch.Tensor
        inputs = {self._input_name: codes}
        self.request.infer(inputs)
        return torch.from_numpy(self.request.get_output_tensor(0).data.copy())


class OVTTSCodeHead:
    """OpenVINO wrapper for TTS code head."""

    def __init__(self, model_path, device):
        self.model = Core().compile_model(model_path, device)
        self.request = self.model.create_infer_request()

    def __call__(self, hidden_states):
        """Get code logits. Returns torch tensor."""
        # OpenVINO directly accepts torch.Tensor
        inputs = {"hidden_states": hidden_states}
        self.request.infer(inputs)
        return torch.from_numpy(self.request.get_output_tensor(0).data.copy())


# ==================== OV LLM with GenerationMixin ====================


class _OVEmbedTokensWrapper:
    """Wrapper to make OVLLMEmbedding callable like model.embed_tokens."""

    def __init__(self, ov_embedding):
        self._ov_embedding = ov_embedding

    def __call__(self, input_ids):
        return self._ov_embedding(input_ids)


class _OVModelWrapper:
    """Wrapper to mimic llm.model with embed_tokens attribute.

    In original: self.llm.model.embed_tokens(input_ids)
    In OV:       self.llm.model.embed_tokens(input_ids)  -> _OVModelWrapper.embed_tokens
    """

    def __init__(self, ov_embedding):
        self.embed_tokens = _OVEmbedTokensWrapper(ov_embedding)


class OVLLMForCausalLM(GenerationMixin):
    """
    OpenVINO wrapper for the LLM component with GenerationMixin support.

    This wraps OVLLMEmbedding + OVLLMLanguageModel into a single class that
    inherits GenerationMixin, mirroring the original Qwen3ForCausalLM.

    Usage in OVMiniCPMO:
        self.llm = OVLLMForCausalLM(model_path, device, config)
        self.llm.model.embed_tokens(input_ids)  # get embeddings
        self.llm.generate(inputs_embeds=..., ...)  # GenerationMixin generate
        self.llm(input_ids=None, inputs_embeds=..., ...)  # forward call

    Follows the pattern of OVQwen3ASRThinkerForConditionalGeneration.
    """

    _is_stateful = False

    def __init__(self, model_path, device, config):
        self.device = torch.device("cpu")  # GenerationMixin expects torch.device
        self._ov_device = device
        self.config = config
        self.dtype = torch.float32

        # Load embedding model
        self._ov_embedding = OVLLMEmbedding(Path(model_path) / LLM_EMBEDDING_NAME, device)

        # Load language model (auto-detects stateful/stateless)
        self._ov_language = OVLLMLanguageModel(Path(model_path) / LLM_LANGUAGE_NAME, device)
        self._is_stateful = self._ov_language._is_stateful
        self.request = self._ov_language.request
        self.input_names = self._ov_language.input_names

        # Mimic original Qwen3ForCausalLM structure: self.model.embed_tokens
        self.model = _OVModelWrapper(self._ov_embedding)

        # Embedding wrapper for GenerationMixin (get_input_embeddings())
        self._embedding_wrapper = self._ov_embedding
        self.get_input_embeddings = lambda: self._embedding_wrapper

        # GenerationMixin required attributes
        self.main_input_name = "input_ids"
        self._supports_flash_attn_2 = True
        self._supports_sdpa = True
        self._supports_static_cache = True
        self._supports_cache_class = False
        self._skip_keys_device_placement = "past_key_values"

        # Load generation config from model_path (saved during export)
        try:
            self.generation_config = GenerationConfig.from_pretrained(str(model_path))
        except Exception:
            self.generation_config = GenerationConfig()

        # State tracking
        self._past_length = 0
        self.next_beam_idx = None
        self.num_pkv = 2

    def can_generate(self):
        """Returns True to validate GenerationMixin.generate() can be used."""
        return True

    def reset_state(self):
        """Reset KV cache state."""
        self._ov_language.reset_state()
        self._past_length = 0
        self.next_beam_idx = None

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for GenerationMixin compatibility.

        Uses past_key_values as sentinel: None means first call (reset state),
        ((),) means subsequent calls (use cached state).

        Returns:
            CausalLMOutputWithPast with logits, past_key_values sentinel, and hidden_states.
        """
        # Get embeddings if not provided
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]

        # Compute position_ids if not provided
        if position_ids is None:
            if past_key_values is None:
                position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            else:
                if cache_position is not None:
                    position_ids = cache_position.unsqueeze(0)
                else:
                    start_pos = self._past_length
                    position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long).unsqueeze(0)

        # Compute attention_mask if not provided
        if attention_mask is None:
            total_len = self._past_length + seq_len if past_key_values is not None else seq_len
            attention_mask = torch.ones([batch_size, total_len], dtype=torch.int64)

        # Reset state on first call (no past_key_values)
        if past_key_values is None:
            self._ov_language.reset_state()
            self.next_beam_idx = np.arange(inputs_embeds.shape[0], dtype=int)
            self._past_length = 0

        # Run LLM forward
        logits, hidden_states = self._ov_language(
            inputs_embeds.to(self.dtype),
            attention_mask,
            position_ids,
        )

        # Update past length
        self._past_length += inputs_embeds.shape[1]

        # Use ((),) as sentinel for stateful KV cache
        past_key_values = ((),)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            # Wrap as tuple: (hidden_states,) so outputs.hidden_states[-1] returns
            # the tensor, matching the original model's layer-tuple format.
            hidden_states=(hidden_states,) if output_hidden_states else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        """Prepare inputs for GenerationMixin.generate().

        First call (past_key_values != ((),)): set past_key_values=None to trigger reset.
        Subsequent calls: pass through with position_ids=None (forward computes it).
        """
        if past_key_values != ((),):
            past_key_values = None
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        model_inputs["position_ids"] = None
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        """Re-order past_key_values for beam search."""
        self.next_beam_idx = np.array(beam_idx)
        return past_key_values

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        return self._past_length


# ==================== OVMiniCPMO Main Inference Class ====================


class OVMiniCPMO:
    """
    OpenVINO implementation of MiniCPM-o-4_5 multimodal model.
    Supports text, image, and audio inputs with speech synthesis output.

    Mirrors the original MiniCPMO class structure:
    - self.llm is an OVLLMForCausalLM (GenerationMixin) that wraps the OV LLM
    - generate() does multimodal preprocessing then calls self.llm.generate()
    - forward() calls self.llm() with inputs_embeds
    - _decode() / _decode_stream() / _decode_text() mirror original methods
    """

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        tts_device: str = None,
        dtype=torch.float32,
    ):
        """
        Initialize OpenVINO MiniCPM-o model.

        Aligned with original MiniCPMO.__init__:
        - self.llm = OVLLMForCausalLM (mirrors Qwen3ForCausalLM with GenerationMixin)
        - self.vpm = self.init_vision_module()
        - self.resampler = self.init_resampler(embed_dim, vision_dim)
        - self.apm = self.init_audio_module()  (returns audio_encoder, audio_projection, audio_avg_pooler)
        - self.tts = self.init_tts_module()
        - self.reset_session(reset_token2wav_cache=True)

        Args:
            model_path: Path to converted OpenVINO models
            device: OpenVINO device for LLM/vision (CPU, GPU, NPU)
            tts_device: OpenVINO device for TTS (defaults to device)
            dtype: Computation dtype
        """
        self.model_path = Path(model_path)
        self.device = torch.device("cpu")
        self._ov_device = device
        self.tts_device = tts_device or device
        self.dtype = dtype

        # Load processor/tokenizer/config (aligned with original prepare_processor)
        self.prepare_processor()

        # LLM (aligned with original: self.llm = Qwen3ForCausalLM(config))
        self.llm = OVLLMForCausalLM(self.model_path, self._ov_device, self.config)
        self.llm.dtype = self.dtype
        self.embed_dim = self.config.hidden_size

        # Also keep direct references for duplex mode compatibility
        self.llm_embedding = self.llm._ov_embedding

        # Init vision module (aligned with original: self.vpm = self.init_vision_module())
        self.vpm = self.init_vision_module()
        self.vision_dim = getattr(self.config, "vision_config", None)
        if self.vision_dim is not None and hasattr(self.vision_dim, "hidden_size"):
            self.vision_dim = self.vision_dim.hidden_size
        else:
            self.vision_dim = self.embed_dim

        # Init resampler (aligned with original: self.resampler = self.init_resampler(...))
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)

        # Init audio module (aligned with original: self.apm = self.init_audio_module())
        self.apm, self.audio_avg_pooler, self.audio_projection_layer = self.init_audio_module()
        # Keep backward-compat aliases
        self.audio_encoder = self.apm
        self.audio_projection = self.audio_projection_layer

        # Init TTS module (aligned with original: self.tts = self.init_tts_module())
        self.tts = self.init_tts_module()

        # Terminators (aligned with original MiniCPMO)
        self.terminators = ["<|im_end|>", "<|endoftext|>"]

        # Think string (aligned with original)
        self.think_str = "<think>\n\n</think>\n\n"

        # Token2wav (initialized later via init_tts)
        self.token2wav = None
        self.token2wav_cache = None

        # Streaming audio processing constants (aligned with original)
        self.SAMPLE_RATE = 16000
        self.CHUNK_MS = 1000
        self.FIRST_CHUNK_MS = 1035
        self.CNN_REDUNDANCY_MS = 0

        # Reset session (aligned with original: self.reset_session(reset_token2wav_cache=True))
        self.reset_session(reset_token2wav_cache=True)

        print("✅ All OpenVINO models loaded successfully!")

    def prepare_processor(self, processor=None, tokenizer=None):
        """Load/prepare processor and tokenizer (aligned with original MiniCPMO.prepare_processor).

        Args:
            processor: Optional pre-loaded processor
            tokenizer: Optional pre-loaded tokenizer
        """
        from transformers import AutoProcessor, AutoTokenizer

        model_path = str(self.model_path)

        if processor is not None:
            self.processor = processor
        else:
            try:
                self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                print(f"✅ Processor loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load processor: {e}")
                self.processor = None

        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif self.processor is not None and hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                print(f"✅ Tokenizer loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load tokenizer: {e}")
                self.tokenizer = None

        # Load config
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Load generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception:
            self.generation_config = GenerationConfig()

    def init_vision_module(self):
        """Initialize vision module (aligned with original MiniCPMO.init_vision_module).

        Returns:
            OVVisionModel wrapping the OpenVINO vision encoder.
        """
        vision = OVVisionModel(self.model_path / VISION_NAME, self._ov_device)
        print("  ✅ Vision Model loaded")
        return vision

    def init_resampler(self, embed_dim, vision_dim):
        """Initialize resampler (aligned with original MiniCPMO.init_resampler).

        Args:
            embed_dim: LLM hidden size
            vision_dim: Vision encoder hidden size

        Returns:
            OVResampler wrapping the OpenVINO resampler.
        """
        resampler = OVResampler(self.model_path / RESAMPLER_NAME, self._ov_device, embed_dim=embed_dim, max_size=(70, 70))
        print("  ✅ Resampler loaded")
        return resampler

    def init_audio_module(self):
        """Initialize audio module (aligned with original MiniCPMO.init_audio_module).

        Original creates:
            self.apm = MiniCPMWhisperEncoder(config.audio_config)
            self.audio_avg_pooler = nn.AvgPool1d(...)
            self.audio_projection_layer = MultiModalProjector(...)

        Returns:
            tuple: (audio_encoder, audio_avg_pooler, audio_projection)
        """
        audio_encoder = OVAudioEncoder(self.model_path / AUDIO_ENCODER_NAME, self._ov_device)
        print("  ✅ Audio Encoder loaded")

        audio_projection = OVAudioProjection(self.model_path / AUDIO_PROJECTION_NAME, self._ov_device)
        print("  ✅ Audio Projection loaded")

        pool_step = getattr(self.config, "audio_pool_step", 5)
        audio_avg_pooler = torch.nn.AvgPool1d(kernel_size=pool_step, stride=pool_step, ceil_mode=True)

        return audio_encoder, audio_avg_pooler, audio_projection

    def init_tts_module(self):
        """Initialize TTS module (aligned with original MiniCPMO.init_tts_module).

        Original creates: MiniCPMTTS(config=self.config.tts_config, audio_tokenizer=None)

        Returns:
            OVTTSModel wrapping all TTS OV submodels.
        """
        tts_embedding = OVTTSEmbedding(self.model_path / TTS_EMBEDDING_NAME, self.tts_device)
        tts_llm = OVTTSLanguageModel(self.model_path / TTS_LANGUAGE_NAME, self.tts_device)
        tts_projector_spk = OVTTSProjectorSpk(self.model_path / TTS_PROJECTOR_SPK_NAME, self.tts_device)
        tts_projector_semantic = OVTTSProjectorSemantic(self.model_path / TTS_PROJECTOR_SEMANTIC_NAME, self.tts_device)
        tts_code_embedding = OVTTSCodeEmbedding(self.model_path / TTS_CODE_EMBEDDING_NAME, self.tts_device)
        tts_code_head = OVTTSCodeHead(self.model_path / TTS_CODE_HEAD_NAME, self.tts_device)
        print("  ✅ TTS Models loaded")

        tts = OVTTSModel.__new__(OVTTSModel)
        tts.ov_model = self
        tts.embedding = tts_embedding
        tts.llm = tts_llm
        tts.projector_spk = tts_projector_spk
        tts.projector_semantic = tts_projector_semantic
        tts.code_embedding = tts_code_embedding
        tts.code_head = tts_code_head

        # Store TTS config (aligned with original self.tts.config)
        tts_config = getattr(self.config, "tts_config", None)
        if tts_config is not None:
            tts.config = tts_config
        else:
            # Fallback
            from types import SimpleNamespace

            tts.config = SimpleNamespace(
                num_audio_tokens=6562,
                num_vq=1,
                audio_bos_token_id=151687,
                audio_tokenizer_type="s3tokenizer_step_audio",
            )

        # audio_tokenizer (initialized later via init_tts)
        tts.audio_tokenizer = None

        return tts

    def init_tts(self, streaming=False, model_dir=None, enable_float16=False, n_timesteps=10, hift_input_len=0):
        """Initialize TTS audio tokenizer (aligned with original MiniCPMO.init_tts).

        For OV, creates OVToken2wav instead of the original Token2wav/CosyVoice.

        Args:
            streaming: Whether to use streaming tokenizer
            model_dir: Path to token2wav model directory
            enable_float16: Whether to use float16
            n_timesteps: Number of diffusion steps

        Returns:
            The audio tokenizer (OVToken2wav)
        """
        if model_dir is None:
            # Try to find token2wav models in current model directory (OV models)
            # First check if we have OV flow models in the main model directory
            ov_flow_emb = Path(self.model_path) / FLOW_EMBEDDINGS_NAME
            if ov_flow_emb.exists():
                # OV models are in the main model directory
                model_dir = self.model_path
            else:
                # Try original model's assets/token2wav path
                original_model_path = str(self.model_path).replace("-OV", "")
                model_dir = os.path.join(original_model_path, "assets", "token2wav")
                if not os.path.exists(model_dir):
                    print(f"⚠️ token2wav model directory not found: {model_dir}")
                    return None

        self.token2wav = OVToken2wav(
            model_dir=model_dir,
            device=self.tts_device,
            float16=enable_float16,
            n_timesteps=n_timesteps,
            hift_input_len=hift_input_len,  # dynamic shape for streaming mel+cache (50+8=58 frames)
            # flow_emb_token_len=50,
            # flow_emb_prompt_len=200,
        )
        self.tts.audio_tokenizer = self.token2wav
        # Force audio_tokenizer_type (aligned with original init_tts line 258)
        self.tts.config.audio_tokenizer_type = "s3tokenizer_step_audio"
        return self.token2wav

    def init_token2wav_cache(self, prompt_speech_16k):
        """Initialize token2wav cache from prompt audio (aligned with original).

        Original (modeling_minicpmo.py line 1346): calls set_stream_cache() to initialize
        streaming flow cache and hift cache, then stores deep copies as base caches.

        Args:
            prompt_speech_16k: Path to reference audio file, or numpy array of 16kHz audio
        """
        if self.token2wav is not None:
            import tempfile
            import soundfile as sf

            # If input is ndarray, save to temp file first (aligned with original)
            if isinstance(prompt_speech_16k, np.ndarray):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    prompt_wav_path = tmp_wav.name
                    sf.write(prompt_wav_path, prompt_speech_16k, 16000)
            else:
                prompt_wav_path = str(prompt_speech_16k)

            # Aligned with original: use set_stream_cache to initialize streaming caches
            self.token2wav.cache = None
            flow_cache_base, hift_cache_base = self.token2wav.set_stream_cache(prompt_wav_path)

            def _clone_recursive(obj):
                """Deep clone nested containers of torch.Tensors."""
                if torch.is_tensor(obj):
                    return obj.clone()
                elif isinstance(obj, dict):
                    return {k: _clone_recursive(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_clone_recursive(v) for v in obj]
                elif isinstance(obj, tuple):
                    return tuple(_clone_recursive(v) for v in obj)
                return obj

            self.token2wav_cache = {
                "hift_cache_base": _clone_recursive(hift_cache_base),
            }
            # flow_cache_base is None for OV (not used), but store if available
            if flow_cache_base is not None:
                self.token2wav_cache["flow_cache_base"] = _clone_recursive(flow_cache_base)

    def init_streaming_processor(self):
        """Initialize streaming processor (aligned with original MiniCPMO.init_streaming_processor)."""
        self.prepare_processor(processor=self.processor, tokenizer=self.tokenizer)

        if hasattr(self.processor, "set_streaming_mode"):
            self.processor.set_streaming_mode(
                mode="exact",
                chunk_ms=self.CHUNK_MS,
                first_chunk_ms=self.FIRST_CHUNK_MS,
                cnn_redundancy_ms=self.CNN_REDUNDANCY_MS,
                enable_sliding_window=True,
                slide_trigger_seconds=30.0,
                slide_stride_seconds=10.0,
            )
            self.processor.reset_streaming()
            self.audio_chunk_idx = 0

    @torch.no_grad()
    def streaming_prefill(
        self,
        session_id,
        msgs,
        omni_mode=True,
        max_slice_nums=None,
        use_tts_template=True,
        enable_thinking=False,
        is_last_chunk=False,
        tokenizer=None,
        processor=None,
        **kwargs,
    ):
        """Streaming prefill — process one message and extend KV cache.

        Aligned with original MiniCPMO.streaming_prefill().

        Args:
            session_id: Session identifier (new session resets state)
            msgs: List of ONE message dict (system/user/assistant)
            omni_mode: Join content with "" (True) or "\\n" (False)
            max_slice_nums: Max image slices
            use_tts_template: Use TTS chat template
            enable_thinking: Enable thinking/CoT
            is_last_chunk: If True, marks the last audio chunk
        """
        from copy import deepcopy
        from PIL import Image

        assert session_id is not None, "session_id cannot be None"
        self.is_first = self.session_id is None or session_id != self.session_id

        if tokenizer is None:
            tokenizer = self.tokenizer
        if processor is None:
            processor = self.processor
        self.prepare_processor(processor=processor, tokenizer=tokenizer)

        images = []
        audios = []

        assert len(msgs) == 1
        copy_msgs = deepcopy(msgs)
        msg = copy_msgs[0]

        assert msg["role"] in ["system", "user", "assistant"]
        is_not_system_prefill = msg["role"] != "system"

        content = msg["content"]
        if isinstance(content, str):
            content = [content]
        cur_msgs = []
        for c in content:
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("<image>./</image>")
            elif isinstance(c, np.ndarray):
                audios.append(c)
                cur_msgs.append("<audio>./</audio>")
            elif isinstance(c, str):
                cur_msgs.append(c)

        cur_contents = "".join(cur_msgs) if omni_mode else "\n".join(cur_msgs)

        if msg["role"] in ["system", "assistant"]:
            self.new_user_msg = True
            self.audio_past_key_values = None

        if self.is_first:
            self.reset_session(reset_token2wav_cache=False)
            self.session_id = session_id
            self.init_streaming_processor()

            if msg["role"] == "user":
                prompt = "<|im_start|>user\n" + cur_contents
                self.new_user_msg = False
            else:
                msg["content"] = cur_contents
                prompt = processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    use_tts_template=use_tts_template,
                    enable_thinking=enable_thinking,
                )
            add_special_tokens = True
        else:
            if self.new_user_msg and msg["role"] == "user":
                if self.llm_generated:
                    if self.llm_generate_completed:
                        prompt = "<|im_end|>\n<|im_start|>user\n" + cur_contents
                    else:
                        prompt = "<|tts_eos|><|im_end|>\n<|im_start|>user\n" + cur_contents
                else:
                    prompt = "<|im_start|>user\n" + cur_contents
                self.new_user_msg = False
            else:
                prompt = cur_contents
            add_special_tokens = False

        # Pad first audio chunk if needed
        if is_not_system_prefill and len(audios) > 0 and self.audio_chunk_idx == 0:
            assert len(audios) == 1, f"streaming mode only supports single audio, currently {len(audios)}"
            first_chunk_samples = int(self.FIRST_CHUNK_MS * self.SAMPLE_RATE / 1000)
            if len(audios[0]) < first_chunk_samples:
                pad_len = first_chunk_samples - len(audios[0])
                audios[0] = np.concatenate([np.zeros(pad_len, dtype=audios[0].dtype), audios[0]])

        model_inputs = processor(
            [prompt],
            [images],
            [audios],
            max_slice_nums=1 if max_slice_nums is None else max_slice_nums,
            use_image_id=False,
            chunk_input=True,
            return_tensors="pt",
            max_length=None,
            sampling_rate=16000,
            add_special_tokens=add_special_tokens,
            online_streaming=is_not_system_prefill,
            audio_chunk_idx=self.audio_chunk_idx,
            is_last_chunk=is_last_chunk,
        )

        if len(audios) > 0 and is_not_system_prefill:
            self.audio_chunk_idx += 1

        # Get embeddings
        model_inputs["inputs_embeds"], _ = self.get_vllm_embedding(model_inputs)
        inputs_embeds = self.get_omni_embedding(
            model_inputs,
            input_embeddings=model_inputs["inputs_embeds"],
            stream_input=is_not_system_prefill,
        )

        # Build attention mask for accumulated KV cache
        seq_len = inputs_embeds.shape[1]
        past_length = self.llm._past_length
        attention_mask = torch.ones((1, past_length + seq_len), dtype=torch.long, device=self.device)

        # Determine past_key_values sentinel
        pkv = None if self.is_first else ((),)

        # Run LLM forward (prefill only, no generation)
        outputs = self.llm(
            past_key_values=pkv,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        # Store the sentinel for subsequent calls
        self.llm_past_key_values = outputs.past_key_values

        return prompt

    @torch.inference_mode()
    def streaming_generate(
        self,
        session_id,
        generate_audio=True,
        use_tts_template=True,
        enable_thinking=False,
        do_sample=True,
        max_new_tokens=256,
        tokenizer=None,
        processor=None,
        **kwargs,
    ):
        """Streaming generate — yield text/audio chunks from accumulated KV cache.

        Aligned with original MiniCPMO.streaming_generate():
        - LLM generates text in chunks of 10 via ChunkPrefillChunkGenerate-like logic
        - For each chunk, constructs TTS condition: emb_text(token_ids) + projector_semantic(F.normalize(hidden_states))
        - TTS backbone generates audio tokens autoregressively (chunk_size=25)
        - Audio tokens buffered and fed to token2wav.stream() for waveform synthesis
        - Yields (waveform_chunk: Tensor, text_chunk: str) pairs

        Args:
            session_id: Session identifier
            generate_audio: If True, yield (wav_chunk, text_chunk); if False, yield (text_chunk, is_finished)
            use_tts_template: Use TTS template
            enable_thinking: Enable thinking mode
            do_sample: Use sampling
            max_new_tokens: Max tokens to generate

        Yields:
            If generate_audio: (wav_chunk: Tensor, text_chunk: str)
            If not generate_audio: (text_chunk: str, is_finished: bool)
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
        if processor is None:
            processor = self.processor

        self.new_user_msg = True
        self.llm_generated = True
        self.llm_generate_completed = False
        self.audio_past_key_values = None

        # Build BOS input for generation
        think_str = getattr(self, "think_str", "")
        if think_str:
            think_str = think_str.replace("\\n", "\n")
        bos_input = "".join(
            [
                "<|im_end|>\n<|im_start|>assistant\n",
                "" if enable_thinking else think_str,
                "<|tts_bos|>" if use_tts_template else "",
            ]
        )

        bos_input_ids = tokenizer.encode(bos_input)
        bos_input_ids = torch.tensor(bos_input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        bos_embeds = self.llm.get_input_embeddings()(bos_input_ids)

        # Terminators for text generation
        terminators_str = getattr(self, "terminators", ["<|tts_eos|>", "<|im_end|>", "</s>"])
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in terminators_str]

        # Forbidden tokens (aligned with original ChunkPrefillChunkGenerate)
        forbidden_tokens = [
            ":",
            "：",
            "；",
            "#",
            "\u201c",
            "\u201d",
            "\u2018",
            "\u2019",
            "@",
            "*",
            "【",
            "】",
            "「",
            "」",
            "(",
            ")",
            "（",
            "）",
            "[",
            "]",
            "&",
            "/",
            "$",
        ]
        forbidden_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in forbidden_tokens]
        bad_token_ids = getattr(tokenizer, "bad_token_ids", [])
        if bad_token_ids:
            forbidden_token_ids.extend(bad_token_ids)

        # Sampling parameters
        temperature_llm = kwargs.get("temperature", 0.7)
        top_p_llm = kwargs.get("top_p", 0.8)
        top_k_llm = kwargs.get("top_k", 100)
        rep_penalty_llm = kwargs.get("repetition_penalty", 1.05)
        length_penalty = kwargs.get("length_penalty", 1.0)

        # ============================================================
        # Inner generator: chunk-based LLM text + TTS audio tokens
        # Yields (audio_token_chunk, is_last_audio_chunk) pairs
        # ============================================================
        def audio_chunk_generator():
            generate_chunk_size = 10
            audio_token_chunk_size = 25  # s3tokenizer 1s = 25 tokens

            generation_inputs_embeds = bos_embeds
            generated_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)
            num_chunks_decode = (max_new_tokens + generate_chunk_size - 1) // generate_chunk_size

            # Track generated token IDs on self for incremental text access (aligned with original)
            self._streaming_generated_token_ids = []

            # ---- TTS state (for generate_audio) ----
            if generate_audio:
                tts = self.tts
                num_audio_tokens = tts.config.num_audio_tokens
                num_vq = getattr(tts.config, "num_vq", 1)
                tts_temperature = getattr(tts.config, "temperature", 0.8)
                tts_eos_token = num_audio_tokens - 1

                # TTS special token embeddings
                audio_bos_token_id = tts.config.audio_bos_token_id
                text_eos_token_id = getattr(tts.config, "text_eos_token_id", 151692)
                audio_bos_embed = tts.embedding(torch.tensor([audio_bos_token_id], dtype=torch.long)).unsqueeze(0)  # [1, 1, hidden]
                text_eos_embed = tts.embedding(torch.tensor([text_eos_token_id], dtype=torch.long)).unsqueeze(0)  # [1, 1, hidden]

                # TTS LLM state
                tts.llm.reset_state()
                tts_past_length = 0
                all_tts_generated_tokens = []  # all generated audio tokens across chunks
                tts_token_buffer = []  # buffer for yielding in chunk_size batches

                # TTS logits processors (aligned with gen_logits in original)
                from transformers import TopPLogitsWarper, TopKLogitsWarper

                tts_top_p = getattr(tts.config, "tts_top_p", 0.85)
                tts_top_k = getattr(tts.config, "tts_top_k", 25)
                tts_rep_penalty = getattr(tts.config, "tts_repetition_penalty", 1.05)

                # Build logits warpers / processors (aligned with gen_logits())
                logits_warpers = []
                logits_warpers.append(TopPLogitsWarper(tts_top_p, min_tokens_to_keep=3))
                logits_warpers.append(TopKLogitsWarper(tts_top_k, min_tokens_to_keep=3))

                logits_processors = []
                if tts_rep_penalty != 1.0:
                    # Inline CustomRepetitionPenaltyLogitsProcessorRepeat
                    class _RepPenaltyProcessor:
                        def __init__(self, penalty, max_input_ids, past_window):
                            self.penalty = penalty
                            self.max_input_ids = max_input_ids
                            self.past_window = past_window

                        def __call__(self, input_ids, scores):
                            if input_ids.size(1) > self.past_window:
                                input_ids = input_ids.narrow(1, -self.past_window, self.past_window)
                            freq = F.one_hot(input_ids, scores.size(1)).sum(1)
                            if freq.size(0) > self.max_input_ids:
                                freq.narrow(0, self.max_input_ids, freq.size(0) - self.max_input_ids).zero_()
                            alpha = torch.pow(self.penalty, freq)
                            scores = scores.contiguous()
                            inp = scores.multiply(alpha)
                            oth = scores.divide(alpha)
                            con = scores < 0
                            out = torch.where(con, inp, oth)
                            return out

                    logits_processors.append(_RepPenaltyProcessor(tts_rep_penalty, num_audio_tokens, 16))

            # ---- LLM chunk generation outer loop (aligned with original) ----
            for chunk_idx in range(num_chunks_decode):
                is_first_chunk = chunk_idx == 0
                chunk_size = generate_chunk_size + (1 if is_first_chunk else 0)

                # Generate one chunk of text tokens
                finished = False
                current_inputs_embeds = generation_inputs_embeds.clone()
                last_hidden_states_list = []
                input_last_hidden_states_list = []
                generated_tokens = []
                PENALTY_WINDOW_SIZE = 128

                for token_idx in range(chunk_size):
                    if is_first_chunk and token_idx == 0:
                        # First chunk: prefill all bos embeddings
                        model_inputs = {
                            "inputs_embeds": current_inputs_embeds,
                            "past_key_values": self.llm_past_key_values,
                            "use_cache": True,
                            "output_hidden_states": generate_audio,
                        }
                    else:
                        # Subsequent: only the latest generated token
                        model_inputs = {
                            "inputs_embeds": current_inputs_embeds[:, -1:, :],
                            "past_key_values": self.llm_past_key_values,
                            "use_cache": True,
                            "output_hidden_states": generate_audio,
                        }

                    outputs = self.llm(**model_inputs)
                    self.llm_past_key_values = outputs.past_key_values

                    logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32)

                    # Forbid specific tokens
                    if forbidden_token_ids:
                        logits[:, forbidden_token_ids] = float("-inf")

                    # Repetition penalty
                    if rep_penalty_llm != 1.0:
                        if generated_ids.shape[1] > 0 or len(generated_tokens) > 0:
                            if len(generated_tokens) > 0:
                                gen_tok_ids = torch.cat(generated_tokens, dim=1)
                                current_seq = torch.cat([generated_ids[:, -PENALTY_WINDOW_SIZE:], gen_tok_ids], dim=1)
                            else:
                                current_seq = generated_ids[:, -PENALTY_WINDOW_SIZE:]
                            unique_ids = torch.unique(current_seq.squeeze(0))
                            for tid in unique_ids:
                                if logits[0, tid] > 0:
                                    logits[0, tid] = logits[0, tid] / rep_penalty_llm
                                else:
                                    logits[0, tid] = logits[0, tid] * rep_penalty_llm

                    # Length penalty (aligned with original ChunkPrefillChunkGenerate)
                    if length_penalty != 1.0:
                        for eos_id in terminators:
                            if logits[0, eos_id] > 0:
                                logits[0, eos_id] = logits[0, eos_id] / length_penalty
                            else:
                                logits[0, eos_id] = logits[0, eos_id] * length_penalty

                    # Temperature
                    if temperature_llm != 1.0:
                        logits = logits / temperature_llm

                    if do_sample:
                        # Top-k
                        if top_k_llm > 0:
                            tok_k_logits, tok_k_indices = torch.topk(logits, min(top_k_llm, logits.size(-1)))
                            logits_filtered = torch.full_like(logits, float("-inf"))
                            logits_filtered.scatter_(1, tok_k_indices, tok_k_logits)
                            logits = logits_filtered
                        # Top-p
                        if top_p_llm < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_remove = cum_probs > top_p_llm
                            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                            sorted_remove[..., 0] = 0
                            remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
                            logits[remove] = float("-inf")
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)

                    # Collect hidden states for TTS
                    if generate_audio and outputs.hidden_states is not None:
                        hs = outputs.hidden_states[-1]  # Last layer hidden states [1, seq_len, hidden_dim]
                        if is_first_chunk and token_idx == 0:
                            input_last_hidden_states_list.append(hs)
                        else:
                            last_hidden_states_list.append(hs)

                    # Check termination
                    if next_token.item() in terminators:
                        finished = True
                        break

                    generated_tokens.append(next_token)
                    next_token_embed = self.llm.get_input_embeddings()(next_token)
                    current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embed], dim=1)

                # Build chunk output (aligned with original ChunkPrefillChunkGenerate)
                if len(generated_tokens) > 0:
                    chunk_token_ids = torch.cat(generated_tokens, dim=1)
                elif finished:
                    chunk_token_ids = torch.zeros((1, 0), dtype=torch.long, device=self.device)
                else:
                    break  # should not happen

                if generate_audio:
                    if len(last_hidden_states_list) > 0:
                        last_hidden_states = torch.cat(last_hidden_states_list, dim=1)
                    elif finished:
                        last_hidden_states = torch.empty((1, 0, bos_embeds.shape[-1]), device=self.device)
                    else:
                        break

                if chunk_token_ids is None or (chunk_token_ids.shape[1] == 0 and not finished):
                    break

                # Determine yield_chunk_token_ids (aligned with original streaming_generate)
                if is_first_chunk:
                    if finished:
                        yield_chunk_token_ids = chunk_token_ids
                    else:
                        yield_chunk_token_ids = chunk_token_ids[:, :-1]
                elif finished:
                    yield_chunk_token_ids = torch.cat([generated_ids[:, -1:], chunk_token_ids], dim=1)
                else:
                    yield_chunk_token_ids = torch.cat([generated_ids[:, -1:], chunk_token_ids[:, :-1]], dim=1)

                if not generate_audio:
                    # Text-only mode: yield (token_ids, finished)
                    self._streaming_generated_token_ids.extend(yield_chunk_token_ids[0].tolist())
                    yield yield_chunk_token_ids, finished
                else:
                    # Audio mode: construct TTS condition and generate audio tokens
                    # Dense connection: emb_text(token_ids) + projector_semantic(F.normalize(hidden_states))
                    if yield_chunk_token_ids.shape[1] > 0:
                        llm_embeds = tts.embedding(yield_chunk_token_ids)
                        hidden_embeds = tts.projector_semantic(last_hidden_states)
                        if getattr(tts.config, "normalize_projected_hidden", True):
                            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
                        tts_embeds = llm_embeds + hidden_embeds
                    else:
                        tts_embeds = torch.empty((1, 0, getattr(tts.config, "hidden_size", 768)), device=self.device)

                    self._streaming_generated_token_ids.extend(yield_chunk_token_ids[0].tolist())

                    # Add text_eos if text finished
                    if finished:
                        tts_embeds = torch.cat([tts_embeds, text_eos_embed], dim=1)

                    # Always add audio_bos
                    condition = torch.cat([tts_embeds, audio_bos_embed], dim=1)

                    # ---- TTS autoregressive generation for this condition ----
                    condition_length = condition.shape[1]
                    tts_finished = False

                    for t in range(500):  # max 500 audio tokens per condition
                        if t == 0:
                            tts_inputs_embeds = condition
                            tts_pos_ids = torch.arange(tts_past_length, tts_past_length + condition_length, dtype=torch.long).unsqueeze(0)
                        else:
                            last_tok = all_tts_generated_tokens[-1]  # [1, 1]
                            tts_inputs_embeds = tts.code_embedding(last_tok.unsqueeze(-1))  # [1, 1, 1] → [1, 1, hidden]
                            tts_pos_ids = torch.tensor([[tts_past_length + condition_length + t - 1]], dtype=torch.long)

                        # Attention mask: total past KV + current input
                        # t=0: mask = tts_past_length + condition_length
                        # t>0: mask = tts_past_length + condition_length + t
                        tts_attn_mask = torch.ones((1, tts_past_length + condition_length + t), dtype=torch.long)

                        tts_hidden = tts.llm(tts_inputs_embeds, tts_attn_mask, tts_pos_ids)

                        # Get audio token logits via code_head
                        tts_logits = tts.code_head(tts_hidden)  # [1, seq_len, num_audio_tokens, num_vq]
                        tts_logits = tts_logits[:, -1, :num_audio_tokens, 0].float()  # [1, num_audio_tokens]

                        tts_logits = tts_logits / tts_temperature

                        audio_bos = len(all_tts_generated_tokens) == 0 and t == 0

                        if not audio_bos:
                            # Apply logits processors and warpers
                            all_gen_t = torch.cat(all_tts_generated_tokens, dim=1) if all_tts_generated_tokens else torch.empty((1, 0), dtype=torch.long)
                            for proc in logits_processors:
                                tts_logits = proc(all_gen_t, tts_logits)
                            for warp in logits_warpers:
                                tts_logits = warp(all_gen_t, tts_logits)

                        scores = F.softmax(tts_logits, dim=-1)
                        idx_next = torch.multinomial(scores, num_samples=1)  # [1, 1]
                        next_id = idx_next[:, 0:1]  # [1, 1]

                        if next_id.item() == tts_eos_token:
                            tts_finished = True
                        else:
                            all_tts_generated_tokens.append(next_id)
                            tts_token_buffer.append(next_id)

                        # Buffer / yield logic (aligned with TTSStreamingGenerator)
                        if len(tts_token_buffer) == 0:
                            if finished:  # text finished
                                yield torch.empty(1, 0, dtype=torch.long), True
                                break
                            else:
                                break
                        elif len(tts_token_buffer) >= audio_token_chunk_size:
                            batch = torch.cat(tts_token_buffer[:audio_token_chunk_size], dim=1)
                            yield batch, False
                            tts_token_buffer = tts_token_buffer[audio_token_chunk_size:]
                        else:
                            if tts_finished:
                                if finished:
                                    batch = torch.cat(tts_token_buffer, dim=1)
                                    yield batch, True
                                    tts_token_buffer = []
                                    break
                                else:
                                    break
                            else:
                                continue

                    tts_past_length += condition_length + t

                # Update state for next chunk
                generated_ids = torch.cat([generated_ids, chunk_token_ids], dim=1)
                generation_inputs_embeds = current_inputs_embeds

                if finished:
                    break

            # Flush remaining TTS buffer
            if generate_audio and len(tts_token_buffer) > 0:
                batch = torch.cat(tts_token_buffer, dim=1)
                yield batch, True
                tts_token_buffer = []

            if generate_audio:
                yield None, None  # End signal

        # ============================================================
        # Outer loop: token2wav streaming (aligned with original)
        # ============================================================
        audio_gen = audio_chunk_generator()

        if generate_audio:
            # Initialize streaming caches (aligned with original)
            def _clone_recursive(obj):
                if torch.is_tensor(obj):
                    return obj.clone()
                elif isinstance(obj, dict):
                    return {k: _clone_recursive(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_clone_recursive(v) for v in obj]
                elif isinstance(obj, tuple):
                    return tuple(_clone_recursive(v) for v in obj)
                return obj

            if self.token2wav_cache is not None:
                self.token2wav.hift_cache_dict = _clone_recursive(self.token2wav_cache["hift_cache_base"])

            # Pre-insert silence tokens (aligned with original: 3 × 4218 silence tokens)
            buffer = [4218] * 3
            pre_lookahead = 3
            CHUNK_SIZE = 25
            prev_text_len = 0

            for audio_token_chunk, is_last_audio_chunk in audio_gen:
                if audio_token_chunk is None:
                    break

                buffer += audio_token_chunk.reshape(-1).tolist()

                if len(buffer) >= CHUNK_SIZE + pre_lookahead:
                    waveform_chunk = self.token2wav.stream(
                        buffer[: CHUNK_SIZE + pre_lookahead],
                        prompt_wav=None,
                        last_chunk=is_last_audio_chunk,
                        return_waveform=True,
                    )
                    waveform_chunk = torch.from_numpy(waveform_chunk)

                    # Get new text chunk
                    new_text = ""
                    if hasattr(self, "_streaming_generated_token_ids") and self._streaming_generated_token_ids:
                        current_text = tokenizer.decode(self._streaming_generated_token_ids)
                        safe_end = len(current_text)
                        while safe_end > 0 and current_text[safe_end - 1] == "\ufffd":
                            safe_end -= 1
                        safe_text = current_text[:safe_end]
                        new_text = safe_text[prev_text_len:]
                        prev_text_len = len(safe_text)

                    yield waveform_chunk, new_text
                    buffer = buffer[CHUNK_SIZE:]

            # Flush remaining buffer
            if len(buffer) > 0:
                waveform_chunk = self.token2wav.stream(
                    buffer,
                    prompt_wav=None,
                    last_chunk=True,
                    return_waveform=True,
                )
                waveform_chunk = torch.from_numpy(waveform_chunk)

                new_text = ""
                if hasattr(self, "_streaming_generated_token_ids") and self._streaming_generated_token_ids:
                    current_text = tokenizer.decode(self._streaming_generated_token_ids)
                    new_text = current_text[prev_text_len:]

                yield waveform_chunk, new_text
        else:
            # Text-only mode: decode tokens incrementally
            accumulated_token_ids = []
            yielded_text_len = 0

            for token_ids, is_finished in audio_gen:
                if torch.is_tensor(token_ids):
                    accumulated_token_ids.extend(token_ids.reshape(-1).tolist())

                full_decoded = tokenizer.decode(accumulated_token_ids, skip_special_tokens=False)

                if is_finished:
                    new_text = full_decoded[yielded_text_len:]
                    # Clean up TTS end-of-sequence marker
                    if "<|tts_eos|>" in new_text:
                        new_text = new_text.split("<|tts_eos|>")[0]
                    yield new_text, is_finished
                else:
                    new_text = full_decoded[yielded_text_len:]
                    safe_end = len(new_text)
                    while safe_end > 0 and new_text[safe_end - 1] == "\ufffd":
                        safe_end -= 1
                    safe_text = new_text[:safe_end]
                    # Clean up TTS end-of-sequence marker
                    if "<|tts_eos|>" in safe_text:
                        safe_text = safe_text.split("<|tts_eos|>")[0]
                    if safe_text:
                        yielded_text_len += len(safe_text)
                        yield safe_text, False

        self.llm_generate_completed = True

    def _tts_generate_audio(self, token_ids, hidden_states):
        """Generate audio waveform from TTS tokens and hidden states.

        Args:
            token_ids: Text token IDs [1, seq_len]
            hidden_states: LLM hidden states for TTS condition [1, seq_len, hidden_dim]

        Returns:
            Audio waveform tensor or None
        """
        if self.tts is None or self.token2wav is None:
            return None

        try:
            # Project hidden states through TTS projector
            tts_projector = getattr(self, "_ov_tts_projector_0", None)
            if tts_projector is None:
                # Try to find projector in the model
                for name in ["_ov_tts_projector_0", "tts_projector"]:
                    tts_projector = getattr(self, name, None)
                    if tts_projector is not None:
                        break

            if tts_projector is not None:
                condition = tts_projector(hidden_states.to(self.dtype))
            else:
                condition = hidden_states

            # Use TTS language model to generate audio codes
            tts_config = self.tts.config
            num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)

            # Reset TTS state
            if hasattr(self.tts, "_ov_language"):
                self.tts._ov_language.reset_state()

            # Generate audio tokens from condition
            audio_tokens = self._generate_tts_tokens(condition, num_audio_tokens)

            if audio_tokens is not None and len(audio_tokens) > 0:
                # Convert audio tokens to waveform using token2wav
                wav = self.token2wav.offline_inference(audio_tokens)
                return wav

            return None
        except Exception as e:
            print(f"TTS generation error: {e}")
            return None

    def _generate_tts_tokens(self, condition, num_audio_tokens, max_tokens=500):
        """Generate TTS audio tokens from condition embeddings.

        Args:
            condition: Condition embeddings from LLM hidden states
            num_audio_tokens: Vocabulary size for audio tokens
            max_tokens: Maximum audio tokens to generate

        Returns:
            Audio token tensor or None
        """
        try:
            tts_eos_token = num_audio_tokens - 1

            # Get TTS embedding and code embedding
            tts_emb = getattr(self, "_ov_tts_embedding", None)
            tts_code_emb = getattr(self, "_ov_tts_code_embedding", None)
            tts_head = getattr(self, "_ov_tts_head", None)

            if tts_emb is None or tts_code_emb is None or tts_head is None:
                return None

            # Prefill condition
            cond_embeds = condition
            if hasattr(self.tts, "_ov_language"):
                self.tts._ov_language.reset_state()

            # Simple autoregressive TTS generation
            audio_bos_id = getattr(self.tts.config, "audio_bos_token_id", 151687)
            bos_embed = tts_code_emb(torch.tensor([[audio_bos_id]], dtype=torch.long))

            # Concatenate condition + bos
            inputs_embeds = torch.cat([cond_embeds, bos_embed], dim=1)

            generated_tokens = []
            past_length = 0

            for step in range(max_tokens):
                attention_mask = torch.ones((1, past_length + inputs_embeds.shape[1]), dtype=torch.long)
                position_ids = torch.arange(past_length, past_length + inputs_embeds.shape[1], dtype=torch.long).unsqueeze(0)

                logits, _ = self.tts._ov_language(
                    inputs_embeds.to(self.dtype),
                    attention_mask,
                    position_ids,
                )
                past_length += inputs_embeds.shape[1]

                # Get logits for audio tokens only
                audio_logits = logits[:, -1, :num_audio_tokens]
                next_token = torch.argmax(audio_logits, dim=-1)
                token_id = next_token.item()

                if token_id == tts_eos_token:
                    break

                generated_tokens.append(token_id)
                inputs_embeds = tts_code_emb(next_token.unsqueeze(0))

            if generated_tokens:
                return torch.tensor([generated_tokens], dtype=torch.long)
            return None

        except Exception as e:
            print(f"TTS token generation error: {e}")
            return None

    def get_input_embeddings(self):
        """Get input embeddings (aligned with original MiniCPMO.get_input_embeddings).

        Original: return self.llm.get_input_embeddings()
        """
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set input embeddings (aligned with original MiniCPMO.set_input_embeddings)."""
        self.llm._embedding_wrapper = value

    def get_output_embeddings(self):
        """Get output embeddings / lm_head (aligned with original MiniCPMO.get_output_embeddings).

        In OV, there is no separate lm_head — it's fused into the language model.
        Returns None as a placeholder.
        """
        return None

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings (aligned with original MiniCPMO.set_output_embeddings).

        No-op for OV since lm_head is fused.
        """
        pass

    def set_decoder(self, decoder):
        """Set the decoder / LLM (aligned with original MiniCPMO.set_decoder)."""
        self.llm = decoder

    def get_decoder(self):
        """Get the decoder / LLM (aligned with original MiniCPMO.get_decoder)."""
        return self.llm

    def reset_session(self, reset_token2wav_cache=True):
        """Reset all session state (aligned with original MiniCPMO.reset_session).

        Original resets:
            llm_past_key_values, audio_past_key_values, tts_last_turn_tokens,
            llm_generated, llm_generate_completed, new_user_msg, session_id,
            token2wav_cache, streaming state, sliding window state
        """
        # Reset KV cache in stateless mode
        if hasattr(self.llm, "_ov_language"):
            self.llm._ov_language.reset_state()
        if hasattr(self.tts, "_ov_language"):
            self.tts._ov_language.reset_state()

        # Reset audio encoder state (stateful mode)
        if hasattr(self, "apm") and hasattr(self.apm, "reset_state"):
            self.apm.reset_state()

        self.llm_past_key_values = None
        self.audio_past_key_values = None
        self.tts_last_turn_tokens = None
        self.llm_generated = False
        self.llm_generate_completed = False
        self.new_user_msg = True
        self.session_id = None

        if reset_token2wav_cache:
            self.token2wav_cache = None

        # Sliding window state (aligned with original)
        self.streaming_text_preserve = 0
        self.streaming_position_offset = 0
        self._rope_inv_freq_cache = {}
        self._next_round_id = 0
        self._pending_round_id = None
        self._omni_chunk_history = []
        self._round_history = []

    # Backward compatibility alias
    def reset_state(self):
        """Backward compatibility alias for reset_session."""
        self.reset_session(reset_token2wav_cache=True)

    @staticmethod
    def get_sys_prompt(ref_audio=None, mode="default", language="en", ref_audio_max_ms=None):
        """Get system prompt (aligned with original MiniCPMO.get_sys_prompt).

        Args:
            ref_audio: Reference audio (numpy array or file path)
            mode: "default", "omni", "audio_assistant", "audio_roleplay", "voice_cloning"
            language: "en" or "zh"
            ref_audio_max_ms: Max duration of ref audio in ms

        Returns:
            dict: System message with role and content
        """
        if ref_audio is not None:
            if isinstance(ref_audio, str):
                import librosa

                if os.path.isfile(ref_audio):
                    duration = ref_audio_max_ms / 1000.0 if ref_audio_max_ms else None
                    ref_audio, _ = librosa.load(ref_audio, sr=16000, mono=True, duration=duration)
                else:
                    print(f"Could not find {ref_audio}")
                    ref_audio = None
            if ref_audio is not None:
                assert isinstance(ref_audio, np.ndarray), "ref_audio error"

        if mode == "omni":
            if language == "zh":
                sys_prompt = ""
                vc_prompt_prefix = "模仿音频样本的音色并生成新的内容。"
                vc_prompt_suffix = "请用这种声音风格来为用户提供帮助。 请认真、高质量地回复用户的问题。 请用高自然度的方式和用户聊天。"
            else:
                sys_prompt = ""
                vc_prompt_prefix = sys_prompt + "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "As an assistant, you will speak using this voice style."

            if ref_audio is not None:
                return {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                return {"role": "system", "content": [sys_prompt]}
        elif mode == "audio_assistant":
            if language == "zh":
                vc_prompt_prefix = "模仿音频样本的音色并生成新的内容。"
                vc_prompt_suffix = "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "Please assist users while maintaining this voice style. Please answer the user's questions seriously and in a high quality. Please chat with the user in a highly human-like and oral style. You are a helpful assistant developed by ModelBest: MiniCPM-Omni."

            if ref_audio is not None:
                return {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                return {"role": "system", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}
        elif mode == "audio_roleplay":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "假装你是上述音频中的人物，与我进行对话。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "Try to role-play the character based on the audio prompt above."

            if ref_audio is not None:
                return {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                return {"role": "system", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}
        elif mode == "voice_cloning":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."

            if ref_audio is not None:
                return {"role": "system", "content": [vc_prompt_prefix, ref_audio]}
            else:
                raise ValueError("ref_audio can't be None in voice_cloning mode.")
        else:
            sys_prompt = "You are a helpful assistant. You can accept audio and text input and output voice and text."
            return {"role": "system", "content": [sys_prompt]}

    def get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
        """Merge audio embeddings into input embeddings (aligned with original MiniCPMO.get_omni_embedding).

        Args:
            data: Dict with audio_features, audio_feature_lens, audio_bounds, etc.
            input_embeddings: Token embeddings to scatter audio into
            chunk_length: Whisper chunk attention length
            stream_input: Whether using streaming audio

        Returns:
            input_embeddings with audio features scattered in
        """
        if stream_input:
            audio_embeddings = self.get_audio_embedding_streaming(data)
        else:
            audio_embeddings = self.get_audio_embedding(data, chunk_length)

        bs = len(input_embeddings)
        if len(data.get("audio_features", [])) > 0:
            assert len(audio_embeddings) == len(input_embeddings)

            if len(audio_embeddings) > 0:
                audio_bounds = data["audio_bounds"]

                if getattr(self.config, "stream_input", False):
                    # Sequential distribution mode (aligned with original config.stream_input path):
                    # Concatenate all audio embeddings and distribute across bounds sequentially.
                    # This handles cases where the processor merges multiple audio segments into
                    # fewer mel spectrograms (e.g., omni mode with multiple video audio chunks).
                    assert bs == 1, "audio stream_input mode only support batch size 1"
                    for i in range(bs):
                        audio_embs = torch.cat(audio_embeddings[i], dim=0).to(device=input_embeddings.device, dtype=input_embeddings.dtype)
                        audio_start_pos = 0
                        for bound in audio_bounds[i]:
                            audio_len = bound[1] - bound[0]
                            input_embeddings[i, bound[0] : bound[1]] = audio_embs[audio_start_pos : audio_start_pos + audio_len, :]
                            audio_start_pos += audio_len
                else:
                    for i in range(bs):
                        audio_embs = audio_embeddings[i]
                        bounds = audio_bounds[i]
                        for embs, bound in zip(audio_embs, bounds):
                            audio_indices = torch.arange(bound[0], bound[1], dtype=torch.long).to(input_embeddings.device)
                            if embs.shape[0] != len(audio_indices):
                                raise ValueError(
                                    f"Shape mismatch: Trying to assign embeddings of shape {embs.shape} " f"to input indices of length {len(audio_indices)}"
                                )
                            input_embeddings[i, audio_indices] = embs.to(input_embeddings.dtype)

        return input_embeddings

    def _get_feat_extract_output_lengths(self, input_lengths):
        """Calculate audio feature lengths after CNN (aligned with original).

        Whisper encoder uses two Conv1d layers with stride 2, so:
            output_length = (input_length - 1) // 2 + 1
        """
        return (input_lengths - 1) // 2 + 1

    def forward(self, data, **kwargs):
        """Forward pass (aligned with original MiniCPMO.forward).

        Original calls:
            self.llm(input_ids=None, position_ids=position_ids, inputs_embeds=vllm_embedding)
        """
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        vllm_embedding = self.get_omni_embedding(
            data,
            input_embeddings=vllm_embedding,
            chunk_length=getattr(self.config, "audio_chunk_length", -1),
        )

        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs,
        )

    def _decode(self, inputs_embeds, tokenizer, attention_mask, **kwargs):
        """Decode using LLM generate (aligned with original MiniCPMO._decode).

        Original calls:
            self.llm.generate(inputs_embeds=..., pad_token_id=0,
                eos_token_id=terminators, attention_mask=...,
                output_hidden_states=True, return_dict_in_generate=True)
        """
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )
        return outputs

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        """Decode with streaming (aligned with original MiniCPMO._decode_stream).

        Original calls self.llm.generate() in a background thread with TextIteratorStreamer.
        """
        from transformers import TextIteratorStreamer

        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_special_tokens=True)
        generation_config = {
            "inputs_embeds": inputs_embeds,
            "pad_token_id": 0,
            "eos_token_id": terminators,
            "streamer": streamer,
        }
        generation_config.update(kwargs)
        thread = threading.Thread(target=self.llm.generate, kwargs=generation_config)
        thread.start()
        return streamer

    def _decode_text(self, result_ids, tokenizer):
        """Decode token IDs to text (aligned with original MiniCPMO._decode_text)."""
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result))
        return result_text

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        audio_features=None,
        audio_feature_lens=None,
        image_bound=None,
        audio_bounds=None,
        spk_bounds=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        stream=False,
        **kwargs,
    ):
        """Generate response (aligned with original MiniCPMO.generate).

        Orchestrates: get_vllm_embedding() -> get_omni_embedding() -> _decode()/_decode_stream()
        """
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens,
            "image_bound": image_bound,
            "audio_bounds": audio_bounds,
            "spk_bounds": spk_bounds,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["tgt_sizes"] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
            model_inputs["inputs_embeds"] = self.get_omni_embedding(
                model_inputs,
                input_embeddings=model_inputs["inputs_embeds"],
                chunk_length=getattr(self.config, "audio_chunk_length", -1),
            )

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
                outputs = {}
            else:
                outputs = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, **kwargs)
                result = self._decode_text(outputs.sequences, tokenizer)

        return result, outputs

    @torch.inference_mode()
    def chat(
        self,
        msgs,
        image=None,
        tokenizer=None,
        processor=None,
        max_new_tokens=4096,
        min_new_tokens=0,
        do_sample=True,
        max_inp_length=8192,
        max_slice_nums=None,
        use_image_id=None,
        enable_thinking=False,
        use_tts_template=False,
        generate_audio=False,
        output_audio_path=None,
        omni_mode=False,
        stream=False,
        stream_input=False,
        sampling=True,
        **kwargs,
    ):
        """
        High-level chat API — aligned with original MiniCPMO.chat().

        Processes multimodal messages (text, images, audio) and generates a response.

        Args:
            msgs: List of message dicts, e.g.
                [{"role": "user", "content": [image, "Describe this."]}]
                Content items can be: str, PIL.Image, np.ndarray (audio 16kHz mono),
                or OpenAI-format dicts like {"type": "text", "text": "..."},
                {"type": "image_url", "image_url": {"url": "/path/to/img"}}.
            image: Optional PIL.Image prepended to first message.
            tokenizer: Override tokenizer (default: self.tokenizer)
            processor: Override processor (default: self.processor)
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            max_inp_length: Maximum input prompt length
            max_slice_nums: Max image slices (None = auto)
            use_image_id: Whether to use image IDs in prompts
            enable_thinking: Whether to enable thinking/CoT mode
            use_tts_template: Whether to use TTS chat template (auto-set when audio present)
            generate_audio: Whether to generate audio output via TTS
            output_audio_path: Path to save generated audio
            omni_mode: Required for omni inference (video + audio input)
            stream: Whether to stream text output
            stream_input: Whether content was streamed in
            sampling: Legacy alias for do_sample
            **kwargs: Additional generation kwargs (top_p, top_k, temperature, repetition_penalty, ...)

        Returns:
            If stream: TextIteratorStreamer for streaming text
            Otherwise: text string
        """
        from copy import deepcopy
        from PIL import Image

        if tokenizer is None:
            tokenizer = self.tokenizer
        if processor is None:
            processor = self.processor
        if not sampling:
            do_sample = False

        if processor is None:
            raise ValueError("Processor is required for chat(). Please ensure processor is loaded.")

        # --- Handle batched vs single input (aligned with original) ---
        batched = isinstance(msgs[0], list)
        msgs_list = msgs if batched else [msgs]
        images_list = [image] if not batched else [None] * len(msgs_list)

        if batched:
            assert image is None, "Please integrate image to msgs when using batch inference."

        prompts_lists = []
        input_images_list = []
        input_audios_list = []
        audio_parts_list = []

        for cur_image, cur_msgs in zip(images_list, msgs_list):
            if isinstance(cur_msgs, str):
                import json as _json

                cur_msgs = _json.loads(cur_msgs)
            copy_msgs = deepcopy(cur_msgs)

            assert len(copy_msgs) > 0, "msgs is empty"

            # Prepend image to first message if provided separately
            if cur_image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [cur_image, copy_msgs[0]["content"]]

            images = []
            audios = []
            audio_parts = []

            for i, msg in enumerate(copy_msgs):
                content = msg.get("content", [])

                # Normalize content to list of native items
                if isinstance(content, str):
                    content = [content]
                elif not isinstance(content, list):
                    content = [content]

                # Normalize structured content (OpenAI format -> native)
                normalized = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type == "text":
                            normalized.append(item.get("text", ""))
                        elif item_type == "image_url":
                            url_obj = item.get("image_url", {})
                            url = url_obj.get("url", "") if isinstance(url_obj, dict) else url_obj
                            normalized.append(Image.open(url).convert("RGB"))
                        elif item_type == "audio_url":
                            import librosa

                            url_obj = item.get("audio_url", {})
                            url = url_obj.get("url", "") if isinstance(url_obj, dict) else url_obj
                            audio_np, _ = librosa.load(url, sr=16000, mono=True)
                            normalized.append(audio_np)
                        else:
                            normalized.append(str(item))
                    else:
                        normalized.append(item)
                content = normalized

                # Replace images/audio with marker strings (aligned with original chat())
                cur_parts = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_parts.append("<image>./</image>")
                    elif isinstance(c, np.ndarray):  # Audio array
                        audios.append(c)
                        audio_parts.append(i)
                        cur_parts.append("<audio>./</audio>")
                        use_tts_template = True
                    elif isinstance(c, str):
                        cur_parts.append(c)

                # Join content: "" for omni/stream, "\n" otherwise
                if omni_mode or stream_input:
                    msg["content"] = "".join(cur_parts)
                else:
                    msg["content"] = "\n".join(cur_parts)

            # Apply chat template (aligned with original)
            prompt = processor.tokenizer.apply_chat_template(
                copy_msgs,
                tokenize=False,
                add_generation_prompt=True,
                use_tts_template=use_tts_template,
                enable_thinking=enable_thinking,
            )

            prompts_lists.append(prompt)
            input_images_list.append(images)
            input_audios_list.append(audios)
            audio_parts_list.append(audio_parts)

        # --- Call processor (aligned with original processor.__call__ signature) ---
        inputs = processor(
            prompts_lists,
            input_images_list,
            input_audios_list,
            audio_parts_list,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            stream_input=stream_input,
            return_tensors="pt",
            max_length=max_inp_length,
        )

        # Remove image_sizes (not used by generate)
        if hasattr(inputs, "pop"):
            inputs.pop("image_sizes", None)
        elif isinstance(inputs, dict):
            inputs.pop("image_sizes", None)

        # --- Prepare generation config ---
        gen_kwargs = {
            "do_sample": do_sample,
            "top_p": kwargs.pop("top_p", 0.8),
            "top_k": kwargs.pop("top_k", 100),
            "temperature": kwargs.pop("temperature", 0.7),
            "repetition_penalty": kwargs.pop("repetition_penalty", 1.05),
        }
        gen_kwargs.update(kwargs)

        # --- Extract fields from processor output ---
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else inputs["input_ids"]
        pixel_values = inputs.get("pixel_values", [[]]) if hasattr(inputs, "get") else inputs.get("pixel_values", [[]])
        tgt_sizes = inputs.get("tgt_sizes", [[]]) if hasattr(inputs, "get") else inputs.get("tgt_sizes", [[]])
        image_bound = inputs.get("image_bound", [[]]) if hasattr(inputs, "get") else inputs.get("image_bound", [[]])
        attention_mask = inputs.get("attention_mask", None) if hasattr(inputs, "get") else inputs.get("attention_mask", None)
        audio_features = inputs.get("audio_features", []) if hasattr(inputs, "get") else inputs.get("audio_features", [])
        audio_feature_lens = inputs.get("audio_feature_lens", []) if hasattr(inputs, "get") else inputs.get("audio_feature_lens", [])
        audio_bounds = inputs.get("audio_bounds", [[]]) if hasattr(inputs, "get") else inputs.get("audio_bounds", [[]])
        spk_bounds = inputs.get("spk_bounds", [[]]) if hasattr(inputs, "get") else inputs.get("spk_bounds", [[]])

        # Ensure list format for pixel_values / tgt_sizes
        if not isinstance(pixel_values, list):
            pixel_values = [pixel_values]
        if not isinstance(tgt_sizes, list):
            tgt_sizes = [tgt_sizes]

        # --- Generate ---
        result, outputs = self.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            tgt_sizes=tgt_sizes,
            audio_features=audio_features if len(audio_features) > 0 else [],
            audio_feature_lens=audio_feature_lens if len(audio_feature_lens) > 0 else [],
            image_bound=image_bound,
            audio_bounds=audio_bounds,
            spk_bounds=spk_bounds,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            stream=stream,
            **gen_kwargs,
        )

        if stream:
            return result  # TextIteratorStreamer

        answer = result[0] if isinstance(result, list) else result

        # Strip TTS markers from text answer
        if answer and "<|tts_eos|>" in answer:
            answer = answer.split("<|tts_eos|>")[0]

        return answer

    def get_vllm_embedding(self, data):
        """Get vision-language model embeddings (aligned with original MiniCPMO.get_vllm_embedding).

        Original:
            vision_hidden_states = self.get_vision_embedding(data)
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])
            # then scatter vision embeddings into vllm_embedding
            return vllm_embedding, vision_hidden_states
        """
        # Get vision embeddings (aligned with original: first step)
        vision_hidden_states = self.get_vision_embedding(data)

        # Get token embeddings (aligned with original: self.llm.model.embed_tokens)
        vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

        # Convert vision hidden states dtype to match vllm_embedding (aligned with original)
        vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states]

        # Scatter vision embeddings into text embeddings (aligned with original)
        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if cur_vs_hs is not None and len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack([torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
                elif self.training if hasattr(self, "training") else False:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def get_vision_embedding(self, data):
        """Get vision embeddings from images (aligned with original model).

        Position IDs are pre-computed in Python (not inside the OV model) because the
        original SiglipVisionEmbeddings uses boolean-mask indexing (index_put_) that
        traces to ScatterNDUpdate with fixed shapes, failing at inference with dynamic sizes.
        """
        if "vision_hidden_states" not in data:
            dtype = self.dtype
            device = self.device
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []

            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                # Pre-compute position_ids (mirrors SiglipVisionEmbeddings.forward logic)
                num_patches_per_side = self.config.vision_config.image_size // self.config.vision_config.patch_size
                boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
                position_ids = torch.zeros((B, max_patches.item()), dtype=torch.long, device=device)
                for batch_idx in range(B):
                    nb_patches_h = tgt_sizes[batch_idx][0].item()
                    nb_patches_w = tgt_sizes[batch_idx][1].item()
                    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
                    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)
                    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
                    pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
                    n_valid = nb_patches_h * nb_patches_w
                    position_ids[batch_idx, :n_valid] = pos_ids

                vision_batch_size = getattr(self.config, "vision_batch_size", 8)
                all_pixel_values = all_pixel_values.type(dtype)

                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(
                            all_pixel_values[start_idx:end_idx],
                            patch_attn_mask[start_idx:end_idx],
                            position_ids[start_idx:end_idx],
                        )
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(
                        all_pixel_values,
                        patch_attn_mask,
                        position_ids,
                    )
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                dummy_feature = None
                for _ in pixel_values_list:
                    vision_hidden_states.append(dummy_feature)

            data["vision_hidden_states"] = vision_hidden_states

        return data["vision_hidden_states"]

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True):
        """Get audio embeddings from mel spectrogram (aligned with original model)."""
        wavforms = data.get("audio_features", [])  # (bs, 80, frames)
        audio_feature_lens_raw = data.get("audio_feature_lens", [])

        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # Create padding mask
            seq_range = (
                torch.arange(
                    0,
                    max_seq_len,
                    dtype=audio_feature_lens.dtype,
                    device=audio_feature_lens.device,
                )
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(batch_size, 1, max_seq_len, max_seq_len)
            audio_attention_mask = audio_attention_mask_.to(dtype=self.dtype, device=self.device)
            audio_attention_mask[audio_attention_mask_] = float("-inf")

            # Run through audio encoder (aligned with original: self.apm)
            # Always use use_cache=False for non-streaming inference to get tensor output
            audio_states = self.apm(wavforms.to(self.dtype), audio_attention_mask, use_cache=False)
            audio_embeds = self.audio_projection_layer(audio_states)

            # Average pooling (aligned with original: self.audio_avg_pooler)
            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            # Calculate feature lengths after pooling
            pool_step = getattr(self.config, "audio_pool_step", 5)
            input_lengths_after_cnn = (audio_feature_lens - 1) // 2 + 1
            input_lengths_after_pooling = (input_lengths_after_cnn - pool_step) // pool_step + 1
            num_audio_tokens = input_lengths_after_pooling.to(dtype=torch.int32)

            # Split into list format
            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        elif dummy:
            # Return empty list for no audio
            return []
        else:
            return []

    def get_audio_embedding_streaming(
        self,
        data,
        use_extra_context=False,
        prefix_extra_frames=1,
        suffix_extra_frames=1,
        cnn_min_length=None,
    ):
        """Extract audio embeddings in a streaming manner using cached key-value pairs.

        This method processes incoming audio features incrementally and stores/updates `past_key_values`
        for faster inference on subsequent audio frames. Only supports batch_size=1 for streaming scenarios.

        Structure aligned with original MiniCPMO.get_audio_embedding_streaming:
        1. Check and manage audio_past_key_values cache length
        2. Build attention mask considering past sequence length
        3. APM processing with past_key_values (use_cache=True)
        4. Update self.audio_past_key_values from APM outputs
        5. Projection and pooling

        NOTE: Current OpenVINO APM model limitation:
        - Standard APM model is stateless (no past_key_values I/O)
        - Use use_streaming_apm=True in conversion to enable KV cache support
        - Without streaming APM, this method works but without KV cache optimization

        Args:
            data (dict):
                - "audio_features" (torch.FloatTensor): Mel-spectrograms of shape (batch_size, 80, frames)
                - "audio_feature_lens" (List[List[int]]): Lengths of each audio segment
            use_extra_context (bool): If True, assumes input contains extra frames for CNN context
            prefix_extra_frames (int): Number of prefix extra frames
            suffix_extra_frames (int): Number of suffix extra frames
            cnn_min_length (int): Minimum length for CNN input padding

        Returns:
            List[List[torch.Tensor]]: Audio embeddings
        """
        wavforms = data.get("audio_features", [])
        audio_feature_lens_raw = data.get("audio_feature_lens", [])

        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            assert batch_size == 1, "Streaming mode only supports batch_size=1"
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # Step 1: Manage audio cache (aligned with original lines 579-586)
            # For stateful APM: cache is internal, track via _cached_seq_len
            # For streaming APM: cache is in self.audio_past_key_values
            apm_is_stateful = hasattr(self.apm, "_is_stateful") and self.apm._is_stateful

            if apm_is_stateful:
                cache_length = self.apm.get_cached_seq_len()
                apm_max_len = 1500
                if cache_length + max_seq_len >= apm_max_len:
                    import logging

                    logging.warning(f"audio cache length {cache_length + max_seq_len} exceed {apm_max_len}, reset.")
                    self.apm.reset_state()
                    cache_length = 0
            else:
                if self.audio_past_key_values is not None:
                    try:
                        cache_length = self.audio_past_key_values[0][0].shape[2]
                        apm_max_len = 1500
                        if cache_length + max_seq_len >= apm_max_len:
                            import logging

                            logging.warning(f"audio_past_key_values length {cache_length + max_seq_len} exceed {apm_max_len}, reset.")
                            self.audio_past_key_values = None
                            cache_length = 0
                    except Exception:
                        self.audio_past_key_values = None
                        cache_length = 0
                else:
                    cache_length = 0

            # Step 2: Build attention mask (aligned with original lines 589-609)
            batch_size, _, max_mel_seq_len = wavforms.shape
            current_seq_len = (max_mel_seq_len - 1) // 2 + 1

            if use_extra_context:
                prefix_to_remove = (prefix_extra_frames + 1) // 2 if prefix_extra_frames > 0 else 0
                suffix_to_remove = (suffix_extra_frames + 1) // 2 if suffix_extra_frames > 0 else 0
                final_seq_len = current_seq_len - prefix_to_remove - suffix_to_remove
            else:
                prefix_to_remove = 0
                suffix_to_remove = 0
                final_seq_len = current_seq_len

            # Calculate total_seq_len from cache length
            past_len = cache_length
            total_seq_len = past_len + current_seq_len

            # Create bidirectional attention mask
            audio_attention_mask = torch.zeros(
                (batch_size, 1, current_seq_len, total_seq_len),
                dtype=self.dtype,
                device=self.device,
            )

            # Step 3: APM processing
            if apm_is_stateful:
                # Stateful APM: cache is internal, just call with attention mask
                audio_states = self.apm(
                    wavforms.to(self.dtype),
                    audio_attention_mask,
                    use_cache=True,
                )
            elif hasattr(self.apm, "supports_kv_cache") and self.apm.supports_kv_cache:
                # Streaming APM with explicit KV cache
                audio_outputs = self.apm(
                    wavforms.to(self.dtype),
                    audio_attention_mask,
                    past_key_values=self.audio_past_key_values,
                    use_cache=True,
                )
                audio_states = audio_outputs[0]
                self.audio_past_key_values = audio_outputs[1]
            else:
                # Fallback to stateless APM (no KV cache optimization)
                audio_states = self.apm(wavforms.to(self.dtype), audio_attention_mask, use_cache=False)

            # Step 3.5: Trim extra context frames if needed (aligned with original)
            if use_extra_context and (prefix_to_remove > 0 or suffix_to_remove > 0):
                # Trim from sequence dimension (dim=1)
                if suffix_to_remove > 0:
                    audio_states = audio_states[:, prefix_to_remove:-suffix_to_remove, :]
                else:
                    audio_states = audio_states[:, prefix_to_remove:, :]

            # Step 4: Projection (aligned with original line 633)
            audio_embeds = self.audio_projection_layer(audio_states)

            # Step 5: Pooling (aligned with original lines 635-637)
            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            # Calculate feature lengths after pooling
            pool_step = getattr(self.config, "audio_pool_step", 5)
            input_lengths_after_cnn = (audio_feature_lens - 1) // 2 + 1
            input_lengths_after_pooling = (input_lengths_after_cnn - pool_step) // pool_step + 1
            num_audio_tokens = input_lengths_after_pooling.to(dtype=torch.int32)

            # Split into list format (aligned with original lines 641-649)
            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        else:
            return []

    def embed_tokens(self, input_ids):
        """Get token embeddings (shortcut for self.llm.model.embed_tokens)."""
        return self.llm.model.embed_tokens(input_ids)

    def llm_forward(self, inputs_embeds, attention_mask, position_ids):
        """Run LLM forward pass directly (for duplex mode).

        Calls the underlying OV language model directly (bypassing GenerationMixin).

        Returns:
            tuple: (logits, hidden_states)
        """
        logits, hidden_states = self.llm._ov_language(inputs_embeds.to(self.dtype), attention_mask, position_ids)
        return logits, hidden_states

    def as_duplex(self, device=None, **kwargs):
        """Convert to duplex mode (aligned with original MiniCPMO.as_duplex).

        Uses OVMiniCPMODuplex.from_existing_model(model=self) to create
        duplex instance without reloading models.

        Args:
            device: Override device (not used for OV)
            **kwargs: Duplex parameters (see OVMiniCPMODuplex._default_duplex_params)

        Returns:
            OVMiniCPMODuplex instance wrapping this model
        """
        return OVMiniCPMODuplex.from_existing_model(model=self, **kwargs)


# ==================== Duplex Mode Supporting Classes ====================


@dataclass
class DuplexWindowConfig:
    """Duplex sliding window configuration.

    Sliding window modes:
    - "off": Disable sliding window
    - "basic": Basic sliding window (trigger by cache length)
    - "context": Sliding window with context preservation (trigger by unit count)
    """

    sliding_window_mode: str = "off"  # "off" / "basic" / "context"

    # Basic sliding window parameters
    basic_window_high_tokens: int = 8000  # High watermark
    basic_window_low_tokens: int = 6000  # Low watermark

    # Context sliding window parameters
    context_previous_max_tokens: int = 500
    context_max_units: int = 24

    verify_mode: bool = False


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def get_rotary_cos_sin(
    head_dim: int,
    positions: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    rope_theta: float = 10000.0,
    inv_freq_cache: Optional[Dict[Tuple, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE cos and sin components for given positions."""
    cache_key = (head_dim, device)

    inv_freq = inv_freq_cache.get(cache_key) if inv_freq_cache is not None else None
    if inv_freq is None or inv_freq.device != device or inv_freq.shape[0] != head_dim // 2:
        exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
        inv_freq = 1.0 / (rope_theta**exponent)
        if inv_freq_cache is not None:
            inv_freq_cache[cache_key] = inv_freq

    positions = positions.to(device=device, dtype=torch.float32)
    angles = torch.einsum("i,j->ij", positions, inv_freq)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    cos_full = torch.cat([cos, cos], dim=-1).to(dtype=dtype)
    sin_full = torch.cat([sin, sin], dim=-1).to(dtype=dtype)
    cos_full = cos_full.unsqueeze(0).unsqueeze(0)
    sin_full = sin_full.unsqueeze(0).unsqueeze(0)
    return cos_full, sin_full


def realign_rotary_suffix(
    suffix_keys: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    rope_theta: float = 10000.0,
    inv_freq_cache: Optional[Dict[Tuple, torch.Tensor]] = None,
) -> torch.Tensor:
    """Realign RoPE position encoding after cache eviction."""
    if suffix_keys.numel() == 0:
        return suffix_keys

    head_dim = suffix_keys.shape[-1]
    device = suffix_keys.device
    dtype = suffix_keys.dtype

    cos_old, sin_old = get_rotary_cos_sin(head_dim, old_positions, device, dtype, rope_theta, inv_freq_cache)
    base = cos_old * suffix_keys - sin_old * rotate_half(suffix_keys)

    cos_new, sin_new = get_rotary_cos_sin(head_dim, new_positions, device, dtype, rope_theta, inv_freq_cache)
    return cos_new * base + sin_new * rotate_half(base)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("inf")):
    """Apply top-k and top-p filtering to logits."""
    logits = logits.clone()

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = filter_value

    return logits


class OVStreamDecoder:
    """
    OpenVINO-based StreamDecoder for duplex mode.

    Manages:
    - KV cache with sliding window
    - Unit-based context tracking
    - Token generation and decoding
    - RoPE position offset handling
    """

    def __init__(
        self,
        llm,  # OVLLMLanguageModel
        embedding_fn: Callable,  # Function to embed tokens
        tokenizer,
        hidden_size: int,
        rope_theta: float = 10000.0,
        special_token_ids: Optional[List[int]] = None,
        forbidden_token_ids: Optional[List[int]] = None,
    ):
        """
        Initialize OVStreamDecoder.

        Args:
            llm: OpenVINO LLM model wrapper
            embedding_fn: Function to convert token IDs to embeddings
            tokenizer: Tokenizer instance
            hidden_size: Model hidden dimension
            rope_theta: RoPE base frequency
            special_token_ids: List of special token IDs
            forbidden_token_ids: List of forbidden token IDs
        """
        self.m = llm
        self.embedding_fn = embedding_fn
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.rope_theta = rope_theta
        self.listen_id = tokenizer.eos_token_id

        # Special token IDs
        self.chunk_eos_id = tokenizer.convert_tokens_to_ids("<|chunk_eos|>")
        self.chunk_tts_eos_id = tokenizer.convert_tokens_to_ids("<|chunk_tts_eos|>")
        self.turn_eos_id = tokenizer.convert_tokens_to_ids("<|turn_eos|>")
        self.speak_id = tokenizer.convert_tokens_to_ids("<|speak|>")

        self.special_token_ids = special_token_ids if special_token_ids is not None else []

        # Cache special tokens for filtering
        self._all_special_ids = set()
        self._all_special_tokens_text = set()
        if tokenizer:
            if hasattr(tokenizer, "all_special_ids"):
                self._all_special_ids = set(tokenizer.all_special_ids)
            if hasattr(tokenizer, "all_special_tokens"):
                self._all_special_tokens_text = set(tokenizer.all_special_tokens)

        custom_special_tokens = [
            "<unit>",
            "</unit>",
            "<image>",
            "</image>",
            "<slice>",
            "</slice>",
            "<|listen|>",
            "<|speak|>",
            "<|tts_bos|>",
            "<|tts_eos|>",
            "<|audio_start|>",
            "<|audio_end|>",
            "<|chunk_eos|>",
            "<|chunk_tts_eos|>",
            "<|turn_eos|>",
        ]
        self._all_special_tokens_text.update(custom_special_tokens)
        for token in custom_special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                self._all_special_ids.add(token_id)

        # Forbidden tokens
        if forbidden_token_ids is None:
            self.forbidden_token_ids = []
        elif isinstance(forbidden_token_ids, int):
            self.forbidden_token_ids = [forbidden_token_ids]
        else:
            self.forbidden_token_ids = list(forbidden_token_ids)
        self.forbidden_token_ids.append(self.chunk_eos_id)

        # Initialize state
        self._cache = None  # Internal cache (OpenVINO LLM manages this)
        self.context = ""
        self.generated_tokens = []
        self.generated_special_tokens = []
        self.embeds = None
        self.system_embeds = None

        # Stateful mode detection
        self._is_stateful = getattr(llm, "_is_stateful", False)
        self._cache_length: int = 0  # Track cache length internally for stateful mode

        # Sliding window states
        self._unit_history: List[Dict[str, Any]] = []
        self._next_unit_id: int = 0
        self._pending_unit_id: Optional[int] = None
        self._pending_unit_start_cache_len: int = 0
        self._system_preserve_length: int = 0
        self._position_offset: int = 0
        self._window_config = DuplexWindowConfig()
        self._window_enabled: bool = True
        self._rope_inv_freq_cache: Dict[Tuple, torch.Tensor] = {}

        # Context preserving sliding window states
        self._preserve_prefix_length: int = 0
        self._previous_content_length: int = 0
        self._suffix_token_ids: List[int] = []
        self._previous_marker: str = "\n\nprevious: "
        self._previous_marker_token_ids: List[int] = []
        self._has_previous: bool = False
        self._previous_text: str = ""
        self._previous_token_ids: List[int] = []

        # Statistics
        self._sliding_event_count: int = 0
        self._total_dropped_tokens: int = 0
        self._total_dropped_units: int = 0

        # System prompt template (for context mode)
        self._system_prompt_template: Optional[str] = None

    def reset(self):
        """Reset decoder state."""
        self.context = ""
        self._cache = None
        self.generated_tokens = []
        self.generated_special_tokens = []
        self.embeds = None
        self.system_embeds = None

        # Reset sliding window states
        self._unit_history = []
        self._next_unit_id = 0
        self._pending_unit_id = None
        self._pending_unit_start_cache_len = 0
        self._system_preserve_length = 0
        self._position_offset = 0
        self._rope_inv_freq_cache = {}

        # Reset context preserving states
        self._preserve_prefix_length = 0
        self._previous_content_length = 0
        self._suffix_token_ids = []
        self._previous_marker_token_ids = []
        self._has_previous = False
        self._previous_text = ""
        self._previous_token_ids = []

        # Reset statistics
        self._sliding_event_count = 0
        self._total_dropped_tokens = 0
        self._total_dropped_units = 0

        # Reset cache length tracker
        self._cache_length = 0

        # Reset LLM state
        self.m.reset_state()

    @property
    def cache(self):
        """Access LLM's KV cache for API compatibility with original StreamDecoder.
        For stateful models, returns None (cache is internal to OpenVINO).
        """
        return self.m.past_key_values

    @cache.setter
    def cache(self, value):
        """Set LLM's KV cache. No-op for stateful models."""
        if not self._is_stateful:
            self.m.past_key_values = value

    def get_cache_length(self) -> int:
        """Get current KV cache length.
        - Stateful: uses internal counter (updated by feed())
        - Stateless: queries actual KV cache tensor shape
        """
        if self._is_stateful:
            return self._cache_length
        if self.m.past_key_values is None or len(self.m.past_key_values) == 0:
            return 0
        # past_key_values is a list of (key, value) tuples for each layer
        # key shape: [batch, num_heads, seq_len, head_dim]
        key = self.m.past_key_values[0][0]
        if hasattr(key, "shape"):
            return key.shape[2]  # seq_len dimension
        return 0

    def get_total_generated_tokens(self) -> int:
        """Get total generated tokens from unit history."""
        return sum(len(u.get("generated_tokens", [])) for u in self._unit_history)

    def set_window_config(self, config: DuplexWindowConfig) -> None:
        """Set sliding window configuration."""
        self._window_config = config

    def set_window_enabled(self, enabled: bool) -> None:
        """Enable/disable sliding window."""
        self._window_enabled = enabled

    def embed_token(self, tid: Union[int, torch.Tensor]) -> torch.Tensor:
        """Embed a single token."""
        if isinstance(tid, int):
            tid = torch.tensor([tid], dtype=torch.long)
        return self.embedding_fn(tid)

    def embed_tokens(self, token_ids: List[int]) -> torch.Tensor:
        """Embed multiple tokens."""
        if not token_ids:
            return torch.empty(0, self.hidden_size)
        tids = torch.tensor(token_ids, dtype=torch.long)
        return self.embedding_fn(tids)

    # ============ Unit Management ============

    def register_unit_start(self) -> int:
        """Register start of a new unit."""
        self._pending_unit_id = self._next_unit_id
        self._pending_unit_start_cache_len = self.get_cache_length()
        return self._pending_unit_id

    def register_unit_end(
        self,
        input_type: str,
        generated_tokens: Optional[List[int]] = None,
        is_listen: bool = False,
        generated_text: Optional[str] = None,
    ):
        """Register end of a unit."""
        if self._pending_unit_id is None:
            return

        current_cache_len = self.get_cache_length()
        unit_len = current_cache_len - self._pending_unit_start_cache_len

        if unit_len > 0:
            entry = {
                "unit_id": self._pending_unit_id,
                "length": unit_len,
                "type": input_type,
                "generated_tokens": generated_tokens or [],
                "generated_text": generated_text or "",
                "is_listen": is_listen,
            }
            self._unit_history.append(entry)

        self._pending_unit_id = None
        self._pending_unit_start_cache_len = 0
        self._next_unit_id += 1

    def register_system_prompt(self):
        """Register system prompt preserve length."""
        self._system_preserve_length = self.get_cache_length()

    def register_system_prompt_with_context(
        self,
        suffix_token_ids: Optional[List[int]] = None,
        context_previous_marker: str = "\n\nprevious: ",
    ):
        """Register system prompt with context preservation."""
        self._preserve_prefix_length = self.get_cache_length()
        self._previous_content_length = 0
        self._suffix_token_ids = suffix_token_ids or []
        self._system_preserve_length = self._preserve_prefix_length + len(self._suffix_token_ids)

        self._previous_marker = context_previous_marker
        self._previous_marker_token_ids = self.tokenizer.encode(context_previous_marker, add_special_tokens=False) if self.tokenizer else []
        self._has_previous = False
        self._previous_text = ""
        self._previous_token_ids = []

    # ============ Feed & Decode ============

    @torch.no_grad()
    def feed(self, embeds: torch.Tensor, return_logits: bool = False):
        """
        Feed embeddings to the model (aligned with original StreamDecoder.feed).

        Args:
            embeds: Input embeddings [L, H] or [1, L, H]
            return_logits: Whether to return logits and hidden states

        Returns:
            If return_logits: (logits, hidden_states) where
                - logits: [1, 1, vocab_size] (last token's logits)
                - hidden_states: [1, 1, hidden_size] (last token's hidden state)
            Else: None
        """
        if embeds.dim() == 3:
            embeds = embeds.squeeze(0)

        L = embeds.size(0)
        if L == 0:
            return None if not return_logits else (None, None)

        past_len = self.get_cache_length()
        position_ids = torch.arange(past_len + self._position_offset, past_len + self._position_offset + L, dtype=torch.long).unsqueeze(0)

        attention_mask = torch.ones((1, past_len + L), dtype=torch.long)

        # Run through OpenVINO LLM (matches original StreamDecoder API)
        inputs_embeds = embeds.unsqueeze(0) if embeds.dim() == 2 else embeds

        # OpenVINO LLM forward pass (stateful or stateless)
        logits, hidden_states = self.m(inputs_embeds, attention_mask, position_ids)

        # Update cache length tracker (for stateful mode)
        if self._is_stateful:
            self._cache_length += L

        if return_logits:
            # Return last token's logits and hidden states (matching original behavior)
            return logits[:, -1:], hidden_states[:, -1:] if hidden_states is not None else None
        return None

    @torch.no_grad()
    def decode(
        self,
        logits: torch.Tensor,
        mode: str = "sampling",
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.8,
        listen_top_k: Optional[int] = None,
        listen_prob_scale: float = 1.0,
        text_repetition_penalty: float = 1.05,
        text_repetition_window_size: int = 512,
    ) -> torch.Tensor:
        """
        Decode logits to token ID.

        OPTIMIZATION: Merged double-sampling into single pass.
        Original code did softmax+multinomial twice: once to check for chunk_eos,
        and again after applying rep penalty/top-k/top-p. Now we apply all
        modifications first, sample once, then check the result.

        Args:
            logits: Model output logits
            mode: "sampling" or "greedy"
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            listen_top_k: Force listen if in top-k
            listen_prob_scale: Scale factor for listen token probability
            text_repetition_penalty: Repetition penalty
            text_repetition_window_size: Window size for repetition penalty

        Returns:
            Next token ID tensor
        """
        logits = logits.clone().float()
        if logits.dim() == 3:
            logits = logits[:, -1, :]  # [1, vocab]

        eos_id = self.chunk_eos_id

        # Apply forbidden token filtering (but NOT chunk_eos — we need to detect it)
        if self.forbidden_token_ids:
            for fid in self.forbidden_token_ids:
                if fid != eos_id:
                    logits[:, fid] = float("-inf")

        # Apply repetition penalty
        if text_repetition_penalty != 1.0 and len(self.generated_tokens) > 0:
            recent_tokens = list(set(self.generated_tokens[-text_repetition_window_size:]))
            for token_id in recent_tokens:
                if token_id < logits.size(-1):
                    if text_repetition_penalty > 1.0:
                        logits[0, token_id] /= text_repetition_penalty
                    else:
                        logits[0, token_id] *= 1.0 / text_repetition_penalty

        # Apply listen probability scaling
        if listen_prob_scale != 1.0:
            logits[0, self.listen_id] *= listen_prob_scale

        # Check listen_top_k
        if listen_top_k is not None:
            listen_rank = (logits[0] > logits[0, self.listen_id]).sum().item()
            if listen_rank < listen_top_k:
                next_token_id = torch.tensor([self.listen_id], dtype=torch.long)
                return next_token_id

        # Single-pass sampling/greedy (merged from previous double-sampling)
        if mode == "greedy":
            next_token_id = torch.argmax(logits, dim=-1)
        elif mode == "sampling":
            logits = logits / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            raise ValueError(f"Unsupported decode mode: {mode}")

        # Check for chunk_eos after sampling
        token_val = next_token_id.item() if next_token_id.dim() == 0 else next_token_id[0].item()
        if token_val == eos_id:
            return torch.tensor([eos_id], dtype=torch.long)

        # Track generated tokens
        if token_val not in self.special_token_ids:
            self.generated_tokens.append(token_val)
        else:
            self.generated_special_tokens.append(token_val)

        return next_token_id

    # ============ Sliding Window ============

    def _get_rope_theta(self) -> float:
        """Get model RoPE theta."""
        return self.rope_theta

    def _drop_tokens_from_cache(self, length: int) -> bool:
        """Drop tokens from KV cache (protect system prompt).

        Removes tokens in the range [preserve, preserve + length) from the
        KV cache. The system prompt region [0, preserve) is always kept.
        Uses numpy slicing on the manual KV cache arrays.

        After dropping, position_offset is increased so that subsequent
        position_ids remain monotonically increasing (required for RoPE).

        Args:
            length: Number of tokens to drop from cache.

        Returns:
            True if tokens were successfully dropped, False otherwise.
        """
        # Stateful models cannot have their KV cache sliced externally
        if self._is_stateful:
            print("⚠️ Warning: sliding window not supported with stateful LLM model, skipping cache drop")
            return False

        if self.cache is None or length <= 0:
            return False

        cache_len = self.get_cache_length()
        preserve = self._system_preserve_length

        # Validate: can't drop more than available non-system tokens
        if preserve + length > cache_len:
            length = cache_len - preserve
            if length <= 0:
                return False

        # Slice KV cache: keep [0, preserve) + [preserve+length, cache_len)
        # KV shape: [batch, num_heads, seq_len, head_dim]
        new_cache = []
        for key, value in self.cache:
            # seq_len is at axis 2
            new_key = np.concatenate([key[:, :, :preserve, :], key[:, :, preserve + length :, :]], axis=2)
            new_value = np.concatenate([value[:, :, :preserve, :], value[:, :, preserve + length :, :]], axis=2)
            new_cache.append((new_key, new_value))

        # Update cache (need a copy since we're creating new arrays from slicing)
        self.cache = new_cache

        # Update position offset so RoPE positions remain consistent
        self._position_offset += length

        # Update statistics
        self._total_dropped_tokens += length
        self._sliding_event_count += 1

        return True

    def _drop_next_unit(self) -> bool:
        """Drop the earliest non-system unit."""
        for entry in self._unit_history:
            unit_id = entry.get("unit_id")
            if unit_id is None or entry.get("type") == "system":
                continue

            total_len = entry.get("length", 0)
            if total_len > 0:
                if self._drop_tokens_from_cache(total_len):
                    self._unit_history.remove(entry)
                    return True
        return False

    def enforce_window(self) -> bool:
        """Enforce basic sliding window strategy."""
        if not self._window_enabled:
            return False

        cfg = self._window_config
        cache_len_before = self.get_cache_length()

        if cache_len_before <= cfg.basic_window_high_tokens:
            return False

        dropped_count = 0
        cache_len = cache_len_before
        while cache_len > cfg.basic_window_low_tokens:
            if not self._drop_next_unit():
                break
            dropped_count += 1
            cache_len = self.get_cache_length()

        if dropped_count > 0:
            self._sliding_event_count += 1
            self._total_dropped_tokens += cache_len_before - cache_len
            self._total_dropped_units += dropped_count

        return dropped_count > 0

    def enforce_window_with_context(self) -> bool:
        """Enforce context-preserving sliding window."""
        if not self._window_enabled:
            return False

        cfg = self._window_config
        if cfg.sliding_window_mode != "context":
            return self.enforce_window()

        units_before = len(self._unit_history)
        if units_before <= cfg.context_max_units:
            return False

        # For now, fallback to basic window
        # Full context preservation requires complex cache manipulation
        return self.enforce_window()

    def get_window_stats(self) -> Dict[str, Any]:
        """Get sliding window statistics."""
        unit_lengths = [u["length"] for u in self._unit_history]
        return {
            "cache_length": self.get_cache_length(),
            "unit_count": len(self._unit_history),
            "unit_lengths": unit_lengths,
            "unit_total_length": sum(unit_lengths),
            "system_preserve_length": self._system_preserve_length,
            "position_offset": self._position_offset,
            "window_enabled": self._window_enabled,
            "total_generated_tokens": self.get_total_generated_tokens(),
            "sliding_event_count": self._sliding_event_count,
            "total_dropped_tokens": self._total_dropped_tokens,
            "total_dropped_units": self._total_dropped_units,
        }


class OVTTSModel:
    """Wrapper for TTS model components."""

    def __init__(self, ov_model: OVMiniCPMO):
        self.ov_model = ov_model
        self.embedding = ov_model.tts_embedding
        self.llm = ov_model.tts_llm
        self.projector_spk = ov_model.tts_projector_spk
        self.projector_semantic = ov_model.tts_projector_semantic
        self.code_embedding = ov_model.tts_code_embedding
        self.code_head = ov_model.tts_code_head

    def reset_state(self):
        """Reset TTS state."""
        self.llm.reset_state()

    def forward(self, inputs_embeds, attention_mask, position_ids):
        """Run TTS forward pass."""
        logits = self.llm(inputs_embeds, attention_mask, position_ids)
        return logits


class OVMiniCPMODuplex:
    """
    Duplex mode for MiniCPM-o-4_5 with streaming support.

    Aligned with original MiniCPMODuplex:
    - Created via from_existing_model(model) classmethod (NOT inheritance)
    - Wraps OVMiniCPMO via self.model (same as original self.model = model)
    - streaming_prefill: Process multimodal input (video frames + audio chunks)
    - streaming_generate: Generate response with listen/speak decisions
    - TTS pipeline for audio synthesis

    Uses OVStreamDecoder for token generation and sliding window management.

    Performance Optimizations:
    - Async pipeline support for parallel processing
    - Thread pool for concurrent audio/vision processing
    """

    def __getattr__(self, name):
        """Delegate attribute access to the underlying OVMiniCPMO model.

        This allows calling methods like init_tts(), chat(), get_sys_prompt(),
        reset_session(), init_token2wav_cache() directly on the duplex object,
        which forwards them to self.model (the wrapped OVMiniCPMO instance).
        """
        if name == "model":
            raise AttributeError(name)
        model = self.__dict__.get("model")
        if model is not None:
            return getattr(model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # Default duplex parameters (aligned with original MiniCPMODuplex._default_duplex_params)
    _default_duplex_params = {
        "generate_audio": True,
        "ls_mode": "explicit",
        "max_new_speak_tokens_per_chunk": 20,
        "text_repetition_penalty": 1.05,
        "temperature": 0.7,
        "top_k": 100,
        "top_p": 0.8,
        "text_repetition_window_size": 512,
        "listen_prob_scale": 1.0,
        "force_listen_count": 0,
        "tts_temperature": 0.8,
        "tts_repetition_penalty": 1.05,
        "enable_float16": False,
        "n_timesteps": 10,
        "chunk_ms": 1000,
        "first_chunk_ms": 1035,
        "cnn_redundancy_ms": 20,
        "sample_rate": 16000,
        "sliding_window_mode": "off",
        "basic_window_high_tokens": 8000,
        "basic_window_low_tokens": 6000,
        "context_previous_max_tokens": 500,
        "context_max_units": 24,
    }

    @classmethod
    def from_existing_model(
        cls,
        model: "OVMiniCPMO",
        **kwargs,
    ) -> "OVMiniCPMODuplex":
        """Create OVMiniCPMODuplex from an existing OVMiniCPMO instance.

        Aligned with original MiniCPMODuplex.from_existing_model().

        Args:
            model: Existing OVMiniCPMO instance
            **kwargs: Override default duplex parameters

        Returns:
            OVMiniCPMODuplex instance wrapping the model
        """
        instance = cls.__new__(cls)

        instance.name_or_path = getattr(model.config, "_name_or_path", "")

        # Get default params helper (aligned with original)
        def get_param(name):
            if name in kwargs:
                return kwargs[name]
            return cls._default_duplex_params.get(name)

        instance.generate_audio = get_param("generate_audio")
        instance.ls_mode = get_param("ls_mode")

        # Reuse the existing model - THIS IS THE KEY: no reloading!
        # (aligned with original: instance.model = model)
        instance.model = model
        instance.device = model.device
        instance._ov_device = model._ov_device
        instance.tts_device = model.tts_device
        instance.dtype = model.dtype
        instance.model_path = model.model_path

        instance.processor = model.processor
        instance.tokenizer = model.tokenizer
        instance.config = model.config
        instance.generation_config = model.generation_config

        # Ensure model has processor reference (aligned with original)
        instance.model.processor = instance.processor

        # Initialize TTS token2wav (aligned with original: model.init_tts(streaming=True, ...))
        enable_float16 = get_param("enable_float16")
        n_timesteps = get_param("n_timesteps")
        if instance.model.token2wav is None:
            instance.model.init_tts(
                streaming=True,
                enable_float16=enable_float16,
                n_timesteps=n_timesteps,
            )

        instance.break_event = threading.Event()
        instance.session_stop_event = threading.Event()

        # LLM generation config (aligned with original)
        instance.max_new_speak_tokens_per_chunk = get_param("max_new_speak_tokens_per_chunk")
        instance.text_repetition_penalty = get_param("text_repetition_penalty")
        instance.temperature = get_param("temperature")
        instance.top_k = get_param("top_k")
        instance.top_p = get_param("top_p")
        instance.text_repetition_window_size = get_param("text_repetition_window_size")
        instance.listen_prob_scale = get_param("listen_prob_scale")
        instance.force_listen_count = get_param("force_listen_count")

        # TTS generation config (aligned with original)
        tts_temp_value = get_param("tts_temperature")
        instance.tts_temperature = torch.tensor([tts_temp_value], dtype=torch.float, device=instance.device)
        instance.tts_repetition_penalty = get_param("tts_repetition_penalty")

        # Stream config (aligned with original)
        instance.CHUNK_MS = get_param("chunk_ms")
        instance.FIRST_CHUNK_MS = get_param("first_chunk_ms")
        instance.CNN_REDUNDANCY_MS = get_param("cnn_redundancy_ms")
        instance.SAMPLE_RATE = get_param("sample_rate")

        instance.model.CHUNK_MS = instance.CHUNK_MS
        instance.model.FIRST_CHUNK_MS = instance.FIRST_CHUNK_MS
        instance.model.CNN_REDUNDANCY_MS = instance.CNN_REDUNDANCY_MS
        instance.model.SAMPLE_RATE = instance.SAMPLE_RATE

        # Initialize special tokens (aligned with original)
        instance._init_special_tokens()

        # Initialize decoder (aligned with original: StreamDecoder(llm=instance.model.llm, ...))
        instance._init_decoder()

        # Sliding window config (aligned with original)
        sliding_window_mode = get_param("sliding_window_mode")
        instance.decoder.set_window_config(
            DuplexWindowConfig(
                sliding_window_mode=sliding_window_mode,
                basic_window_high_tokens=get_param("basic_window_high_tokens"),
                basic_window_low_tokens=get_param("basic_window_low_tokens"),
                context_previous_max_tokens=get_param("context_previous_max_tokens"),
                context_max_units=get_param("context_max_units"),
            )
        )
        instance.decoder.set_window_enabled(sliding_window_mode != "off")

        # TTS logits processors (aligned with original)
        instance.tts_logits_processors = None
        instance.tts_eos_token = None
        if instance.generate_audio:
            tts_config = instance.model.tts.config
            num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            instance.tts_logits_processors = (
                gen_logits(
                    num_code=num_audio_tokens,
                    repetition_penalty=instance.tts_repetition_penalty,
                )
                if callable(globals().get("gen_logits", None))
                else None
            )
            instance.tts_eos_token = torch.tensor(
                [num_audio_tokens - 1],
                dtype=torch.long,
                device=instance.device,
            )

        # Initialize streaming state
        instance._reset_streaming_state()

        return instance

    def _init_special_tokens(self):
        """Initialize special token IDs (aligned with original model)."""
        tokenizer = self.tokenizer

        self.eos_token_id = tokenizer.eos_token_id

        # Unit tokens
        self.unit_token_id = tokenizer.convert_tokens_to_ids("<unit>")
        self.unit_end_token_id = tokenizer.convert_tokens_to_ids("</unit>")
        self.image_start_token_id = tokenizer.convert_tokens_to_ids("<image>")
        self.image_end_token_id = tokenizer.convert_tokens_to_ids("</image>")
        self.slice_start_token_id = tokenizer.convert_tokens_to_ids("<slice>")
        self.slice_end_token_id = tokenizer.convert_tokens_to_ids("</slice>")

        # Listen/Speak tokens
        self.listen_token_id = tokenizer.convert_tokens_to_ids("<|listen|>")
        self.speak_token_id = tokenizer.convert_tokens_to_ids("<|speak|>")
        self.tts_bos_token_id = tokenizer.convert_tokens_to_ids("<|tts_bos|>")
        self.tts_eos_token_id = tokenizer.convert_tokens_to_ids("<|tts_eos|>")

        # Chunk tokens
        self.chunk_eos_token_id = tokenizer.convert_tokens_to_ids("<|chunk_eos|>")
        self.chunk_tts_eos_token_id = tokenizer.convert_tokens_to_ids("<|chunk_tts_eos|>")
        self.turn_eos_token_id = tokenizer.convert_tokens_to_ids("<|turn_eos|>")

        # Token sets for termination detection
        self.chunk_terminator_token_ids = [
            self.listen_token_id,
            self.chunk_eos_token_id,
            self.chunk_tts_eos_token_id,
        ]
        self.turn_terminator_token_ids = [self.turn_eos_token_id]
        self.chunk_speak_token_ids = [self.speak_token_id]

        # Forbidden tokens
        self.tts_pad_id = tokenizer.convert_tokens_to_ids("<|tts_pad|>")
        bad_token_ids = getattr(tokenizer, "bad_token_ids", [])
        self.forbidden_token_ids = [self.tts_pad_id] + list(bad_token_ids)

        # TTS config attributes (aligned with original model.tts.config)
        tts_config = getattr(self.config, "tts_config", None)
        if tts_config is not None:
            if isinstance(tts_config, dict):
                self._tts_num_audio_tokens = tts_config.get("num_audio_tokens", 6562)
                self._tts_num_vq = tts_config.get("num_vq", 1)
                self._tts_audio_bos_token_id = tts_config.get("audio_bos_token_id", 151687)
            else:
                self._tts_num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
                self._tts_num_vq = getattr(tts_config, "num_vq", 1)
                self._tts_audio_bos_token_id = getattr(tts_config, "audio_bos_token_id", 151687)
        else:
            self._tts_num_audio_tokens = 6562
            self._tts_num_vq = 1
            self._tts_audio_bos_token_id = 151687

        # TTS EOS token is the last audio token (aligned with original)
        self._tts_eos_token = self._tts_num_audio_tokens - 1

    def _init_decoder(self):
        """Initialize OVStreamDecoder with full functionality."""
        hidden_size = getattr(self.config, "hidden_size", 4096)
        rope_theta = getattr(self.config, "rope_theta", 10000.0)

        # Pass the raw OV language model (not the GenerationMixin wrapper)
        # because OVStreamDecoder calls self.m(embeds, mask, pos) with positional args
        self.decoder = OVStreamDecoder(
            llm=self.model.llm._ov_language,
            embedding_fn=self.model.embed_tokens,
            tokenizer=self.tokenizer,
            hidden_size=hidden_size,
            rope_theta=rope_theta,
            special_token_ids=self.chunk_speak_token_ids + self.chunk_terminator_token_ids,
            forbidden_token_ids=self.forbidden_token_ids,
        )

        # Set window config (aligned with original defaults)
        self.decoder.set_window_config(
            DuplexWindowConfig(
                sliding_window_mode="off",  # Default off, can be enabled
                basic_window_high_tokens=8000,
                basic_window_low_tokens=6000,
                context_previous_max_tokens=500,
                context_max_units=24,
            )
        )

    def _reset_streaming_state(self):
        """Reset all streaming state for new session (aligned with original model)."""
        self.decoder.reset()
        self.model.tts.reset_state()

        # Audio streaming state
        self.audio_chunk_idx = 0
        self.audio_buffer = np.array([], dtype=np.float32)

        # Generation state
        self.total_ids = []
        self.total_hidden = []
        self.res_ids = []
        self.speak_count = 0
        self.pending_logits = None

        # TTS state
        self.tts_past_key_values = None
        self.tts_text_start_pos = 0
        self.tts_current_turn_start_time = None

        # Token2wav state
        self.token2wav_initialized = False
        self.token2wav_buffer = []
        self.pre_lookahead = 3  # default from flow.yaml pre_lookahead_len
        self.hift_cache_base = None  # base HiFT cache for turn resets

        # Turn state
        self.current_turn_ended = True
        self.current_mode = None

        # Force listen control
        self._streaming_generate_count = 0

        # Schema tracking
        self.prefill_schema_tokens = []
        self._current_unit_prefill_tokens = []

        # Session state
        self.session_id = None

        # Clear events
        self.break_event.clear()
        self.session_stop_event.clear()

    def _reset_token2wav_for_new_turn(self):
        """Reset token2wav state for new turn (aligned with original).

        The original resets the flow stream cache and hift cache to
        base values, and prepends silence prefix tokens.
        For OV, we reset hift_cache_dict from base copies and reset the buffer.
        """
        if self.token2wav_initialized:
            # Reset HiFT cache to base values (aligned with original)
            if hasattr(self, "hift_cache_base") and self.hift_cache_base is not None:
                self.model.token2wav.hift_cache_dict = {k: v.clone() for k, v in self.hift_cache_base.items()}
            else:
                self.model.token2wav.hift_cache_dict = {
                    "mel": torch.zeros(1, 80, 0),
                    "source": torch.zeros(1, 1, 0),
                    "speech": torch.zeros(1, 0),
                }
            # Reset buffer with silence prefix (aligned with original: [4218] * 3)
            self.token2wav_buffer = [4218] * 3

    def is_break_set(self) -> bool:
        return self.break_event.is_set()

    def is_session_stop_set(self) -> bool:
        return self.session_stop_event.is_set()

    def set_break_event(self):
        self.break_event.set()

    def clear_break_event(self):
        self.break_event.clear()

    def set_session_stop(self):
        self.session_stop_event.set()

    def clear_session_stop(self):
        self.session_stop_event.clear()

    def interrupt(self):
        """
        Interrupt the model mid-speech and reset to a clean listen state.

        This performs a FULL model-level interruption:
        1. Signals any in-flight operations to abort (break_event)
        2. Injects <|turn_eos|> + </unit> into LLM KV cache to cleanly
           close the old response turn — this is CRITICAL so the model
           sees a proper turn boundary and treats the next input as new
        3. Resets TTS state (KV cache, text position, timing)
        4. Resets token2wav buffer (pending audio codes, flow/hift caches)
        5. Sets current_turn_ended=True so the model can choose to listen
        6. Clears pending logits and accumulated speak state

        The LLM's KV cache (conversation context) is PRESERVED with
        a clean turn boundary injected. The model remembers the conversation
        history but sees the old response as properly ended.

        Call this from the application layer when user voice activity is
        detected during AI speech.
        """
        # 1. Signal any in-flight operations to abort
        self.set_break_event()

        # 2. CRITICAL: Inject turn-end tokens into LLM KV cache
        #    This makes the model see a clean turn boundary:
        #    [old response tokens...] <|turn_eos|> </unit>
        #    Without this, the model's KV cache still has the partial old
        #    response context and it will continue speaking old content.
        try:
            # Feed <|turn_eos|> to signal the response is done
            turn_eos_embed = self.decoder.embed_token(self.turn_eos_token_id)
            self.decoder.feed(turn_eos_embed)
            self.total_ids.append(self.turn_eos_token_id)

            # Feed </unit> to close the unit
            unit_end_embed = self.decoder.embed_token(self.unit_end_token_id)
            self.decoder.feed(unit_end_embed)
            self.total_ids.append(self.unit_end_token_id)

            # Register unit end so sliding window tracking is correct
            if self.decoder._pending_unit_id is not None:
                self.decoder.register_unit_end(
                    input_type="interrupted",
                    generated_tokens=[],
                    is_listen=False,
                    generated_text="[interrupted]",
                )

            print("🔴 [interrupt] Injected <|turn_eos|> + </unit> into LLM KV cache")
        except Exception as e:
            print(f"⚠️ [interrupt] Failed to inject turn-end tokens: {e}")

        # 3. Reset TTS state — stop mid-speech generation
        self.tts_past_key_values = None  # Force TTS KV cache reset on next call
        self.tts_text_start_pos = 0  # Reset TTS text position
        self.tts_current_turn_start_time = None

        # Reset the OV TTS model's internal state (KV cache)
        if hasattr(self.model, "tts") and hasattr(self.model.tts, "reset_state"):
            self.model.tts.reset_state()

        # 4. Reset token2wav buffer — discard pending audio codes
        self._reset_token2wav_for_new_turn()

        # Reset flow/hift caches in token2wav
        if self.model.token2wav is not None:
            if hasattr(self, "hift_cache_base") and self.hift_cache_base is not None:
                self.model.token2wav.hift_cache_dict = {k: v.clone() for k, v in self.hift_cache_base.items()}
            else:
                self.model.token2wav.hift_cache_dict = {
                    "mel": torch.zeros(1, 80, 0),
                    "source": torch.zeros(1, 1, 0),
                    "speech": torch.zeros(1, 0),
                }

        # 5. Allow the model to choose listen on next generate
        self.current_turn_ended = True

        # 6. Clear accumulated speak state for this turn
        self.pending_logits = None
        self.res_ids = []
        self.speak_count = 0

        # 7. Clear break_event so next streaming cycle works normally
        self.clear_break_event()

        print("🔴 [interrupt] Model speech interrupted — turn closed, TTS/token2wav reset, ready to listen")

    def prepare(
        self,
        prefix_system_prompt: Optional[str] = None,
        ref_audio: Optional[np.ndarray] = None,
        prompt_wav_path: Optional[str] = None,
        context_previous_marker: str = "\n\nprevious: ",
        generate_audio: bool = True,
        force_listen_count: int = 0,
        sliding_window_mode: str = "off",
        **kwargs,
    ):
        """
        Prepare for duplex streaming session (aligned with original model).

        Args:
            prefix_system_prompt: System prompt prefix
            ref_audio: Reference audio for TTS voice cloning (16kHz mono)
            prompt_wav_path: Path to reference audio file
            context_previous_marker: Marker for context preservation mode
            generate_audio: Whether to generate audio output
            force_listen_count: Number of initial chunks to force listen
            sliding_window_mode: "off", "basic", or "context"
        """
        prefix_system_prompt = prefix_system_prompt or "Streaming Omni Conversation."

        # Build prefix and suffix
        prefix_system_prompt = "<|im_start|>system\n" + prefix_system_prompt
        suffix_system_prompt = "<|im_end|>"
        if isinstance(ref_audio, np.ndarray):
            prefix_system_prompt += "\n<|audio_start|>"
            suffix_system_prompt = "<|audio_end|>" + suffix_system_prompt

        self.clear_break_event()
        self.clear_session_stop()

        self._reset_streaming_state()
        self.decoder.reset()

        self.generate_audio = generate_audio
        self.force_listen_count = force_listen_count
        self.prompt_wav_path = prompt_wav_path

        # Initialize streaming processor (aligned with original model.init_streaming_processor)
        if self.processor is not None and hasattr(self.processor, "set_streaming_mode"):
            self.processor.set_streaming_mode(
                mode="exact",
                chunk_ms=self.CHUNK_MS,
                first_chunk_ms=self.FIRST_CHUNK_MS,
                cnn_redundancy_ms=self.CNN_REDUNDANCY_MS,
                enable_sliding_window=True,
                slide_trigger_seconds=30.0,
                slide_stride_seconds=10.0,
            )
            self.processor.reset_streaming()

        # Initialize token2wav cache (aligned with original _init_token2wav_cache)
        if prompt_wav_path is not None and prompt_wav_path and generate_audio:
            if self.model.token2wav is not None:
                # Reset and initialize streaming cache (aligned with original)
                self.model.token2wav.reset_cache()
                _, hift_cache = self.model.token2wav.set_stream_cache(prompt_wav_path)
                # Store base copies for turn reset (aligned with original)
                self.hift_cache_base = {k: v.clone() for k, v in hift_cache.items()}
                self.pre_lookahead = 3  # from flow.yaml pre_lookahead_len
                self.token2wav_initialized = True
                # Initialize token2wav buffer with silence prefix (aligned with original)
                self.token2wav_buffer = [4218] * 3

        # Configure sliding window
        self.decoder.set_window_config(
            DuplexWindowConfig(
                sliding_window_mode=sliding_window_mode,
                basic_window_high_tokens=kwargs.get("basic_window_high_tokens", 8000),
                basic_window_low_tokens=kwargs.get("basic_window_low_tokens", 6000),
                context_previous_max_tokens=kwargs.get("context_previous_max_tokens", 500),
                context_max_units=kwargs.get("context_max_units", 24),
            )
        )
        self.decoder.set_window_enabled(sliding_window_mode != "off")

        # Prefill system prompt prefix
        # Prefill system prompt prefix — BATCH optimization:
        # Feed all tokens at once instead of one-by-one to avoid N separate OV infer calls.
        if prefix_system_prompt:
            tokens = self.tokenizer.encode(prefix_system_prompt, add_special_tokens=False)
            if tokens:
                all_embeds = self.decoder.embed_tokens(tokens)  # [N, H]
                self.decoder.feed(all_embeds)

        # Prefill reference audio
        if ref_audio is not None:
            data = self.processor.process_audio([ref_audio]) if self.processor else None
            if data is not None:
                audio_chunk_length = getattr(self.config, "audio_chunk_length", -1)
                embeds_nested = self.model.get_audio_embedding(data, chunk_length=audio_chunk_length)
                embeds = torch.cat([t for g in embeds_nested for t in g], dim=0) if embeds_nested else None
                if embeds is not None:
                    self.decoder.feed(embeds)

        # Register system prompt protection length
        if prefix_system_prompt or suffix_system_prompt or ref_audio is not None:
            if sliding_window_mode == "context":
                # Context preserve mode
                suffix_token_ids = []
                if suffix_system_prompt:
                    suffix_token_ids = self.tokenizer.encode(suffix_system_prompt, add_special_tokens=False)

                # Register (when cache only has prefix, no suffix, no previous)
                self.decoder.register_system_prompt_with_context(
                    suffix_token_ids=suffix_token_ids,
                    context_previous_marker=context_previous_marker,
                )

                # Now feed suffix — BATCH optimization
                if suffix_token_ids:
                    suffix_embeds = self.decoder.embed_tokens(suffix_token_ids)
                    self.decoder.feed(suffix_embeds)
            else:
                # Non-context preserve mode: first feed suffix, then register total length
                if suffix_system_prompt:
                    tokens = self.tokenizer.encode(suffix_system_prompt, add_special_tokens=False)
                    if tokens:
                        suffix_embeds = self.decoder.embed_tokens(tokens)
                        self.decoder.feed(suffix_embeds)
                self.decoder.register_system_prompt()

        full_prompt = ""
        if prefix_system_prompt or suffix_system_prompt:
            if ref_audio is not None:
                full_prompt = (prefix_system_prompt or "") + "[audio embedding]" + (suffix_system_prompt or "")
            else:
                full_prompt = (prefix_system_prompt or "") + (suffix_system_prompt or "")

        print(f"✅ Duplex mode prepared: sliding_window={sliding_window_mode}, ref_audio={'yes' if ref_audio is not None else 'no'}")
        return full_prompt

    @torch.no_grad()
    def streaming_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[List] = None,
        text_list: Optional[List[str]] = None,
        max_slice_nums: Union[int, List[int]] = 1,
        batch_vision_feed: bool = False,
        # Simplex API parameters (dispatched to self.model.streaming_prefill)
        session_id=None,
        msgs=None,
        **kwargs,
    ):
        """
        Streaming prefill with multimodal input (aligned with original model).

        Supports two calling conventions:
        - **Duplex API**: streaming_prefill(audio_waveform=..., frame_list=...)
        - **Simplex API**: streaming_prefill(session_id=..., msgs=...) — delegates to self.model

        Args:
            audio_waveform: Audio waveform (16kHz mono)
            frame_list: List of video frames (PIL Images)
            text_list: List of text segments
            max_slice_nums: Max slices per image
            batch_vision_feed: Whether to batch vision embeddings
            session_id: (Simplex API) Session identifier
            msgs: (Simplex API) List of message dicts

        Returns:
            Dict with success status and timing info (duplex), or prompt string (simplex)
        """
        # Dispatch simplex API calls to the underlying OVMiniCPMO model
        if session_id is not None or msgs is not None:
            return self.model.streaming_prefill(session_id=session_id, msgs=msgs, max_slice_nums=max_slice_nums, **kwargs)

        start_time = time.time()

        def _make_result(success, reason=""):
            return {
                "success": success,
                "reason": reason,
                "cost_all": time.time() - start_time,
            }

        if self.is_session_stop_set() or self.is_break_set():
            return _make_result(False, "Session stopped or break set")

        has_frames = frame_list is not None and len(frame_list) > 0
        has_audio = audio_waveform is not None and len(audio_waveform) > 0
        has_text = text_list is not None and len(text_list) > 0

        if has_frames and has_audio:
            mode = "OMNI"
        elif has_frames:
            mode = "VISION"
        elif has_audio:
            mode = "AUDIO"
        elif has_text:
            mode = "TEXT"
        else:
            return _make_result(False, "No input provided")

        self.pending_logits = None

        # Register unit start for sliding window
        self.decoder.register_unit_start()
        self._current_unit_prefill_tokens = []

        # Feed <unit> token
        self.decoder.feed(self.decoder.embed_token(self.unit_token_id))
        self._current_unit_prefill_tokens.append(self.unit_token_id)

        # Process images
        if has_frames:
            # Normalize max_slice_nums to a list
            if isinstance(max_slice_nums, int):
                max_slice_nums_list = [max_slice_nums] * len(frame_list)
            else:
                max_slice_nums_list = list(max_slice_nums)
                if len(max_slice_nums_list) != len(frame_list):
                    raise ValueError(f"max_slice_nums list length must match frame_list length")

            # Check if all max_slice_nums are the same
            all_same = len(set(max_slice_nums_list)) == 1

            # Prepare vision data
            if self.processor is not None and hasattr(self.processor, "process_image"):
                if all_same:
                    processed = self.processor.process_image(frame_list, max_slice_nums=max_slice_nums_list[0])
                else:
                    # Different max_slice_nums per image
                    all_pixel_values = []
                    all_tgt_sizes = []
                    for frame, max_slices in zip(frame_list, max_slice_nums_list):
                        pf = self.processor.process_image([frame], max_slice_nums=max_slices)
                        all_pixel_values.extend(pf["pixel_values"][0])
                        if hasattr(pf["tgt_sizes"][0], "tolist"):
                            all_tgt_sizes.extend(pf["tgt_sizes"][0].tolist())
                        else:
                            all_tgt_sizes.extend(list(pf["tgt_sizes"][0]))

                    processed = {
                        "pixel_values": [all_pixel_values],
                        "tgt_sizes": [torch.tensor(all_tgt_sizes) if all_tgt_sizes else []],
                    }

                vision_data = {
                    "pixel_values": processed.get("pixel_values", []),
                    "tgt_sizes": processed.get("tgt_sizes", []),
                }
                vision_hidden_states_list = self.model.get_vision_embedding(vision_data)

                if vision_hidden_states_list and len(vision_hidden_states_list) > 0:
                    # Calculate slice counts for each image
                    slice_counts = []
                    for frame_idx, frame in enumerate(frame_list):
                        max_slices = max_slice_nums_list[frame_idx]
                        if hasattr(frame, "size") and hasattr(self.processor, "image_processor"):
                            grid = self.processor.image_processor.get_sliced_grid(frame.size, max_slices, nerver_split=False)
                            if grid is not None:
                                slice_counts.append(1 + grid[0] * grid[1])
                            else:
                                slice_counts.append(1)
                        else:
                            slice_counts.append(1)

                    # Get flattened embeddings
                    all_embeds = vision_hidden_states_list[0]

                    # Collect all feed operations: (embed, is_last_for_vision_mode, token_id_or_none)
                    feed_operations = []

                    embed_idx = 0
                    for img_idx, num_slices in enumerate(slice_counts):
                        if num_slices == 0:
                            continue

                        # Feed source image
                        feed_operations.append((self.decoder.embed_token(self.image_start_token_id), False, self.image_start_token_id))
                        feed_operations.append((all_embeds[embed_idx], False, None))
                        feed_operations.append((self.decoder.embed_token(self.image_end_token_id), False, self.image_end_token_id))
                        embed_idx += 1

                        # Feed HD slices
                        if num_slices > 1:
                            for slice_i in range(1, num_slices):
                                feed_operations.append((self.decoder.embed_token(self.slice_start_token_id), False, self.slice_start_token_id))
                                feed_operations.append((all_embeds[embed_idx], False, None))
                                feed_operations.append((self.decoder.embed_token(self.slice_end_token_id), False, self.slice_end_token_id))
                                embed_idx += 1

                    # Mark last operation for VISION mode
                    if feed_operations:
                        feed_operations[-1] = (feed_operations[-1][0], True, feed_operations[-1][2])

                    # Execute feed operations
                    if batch_vision_feed and feed_operations:
                        # Batch mode: concatenate all embeddings and feed at once
                        all_embeds_list = []
                        for embed, is_last, token_id in feed_operations:
                            if embed.dim() == 1:
                                embed = embed.unsqueeze(0)
                            all_embeds_list.append(embed)

                        all_embeds_to_feed = torch.cat(all_embeds_list, dim=0)

                        if mode == "VISION":
                            self.pending_logits, _ = self.decoder.feed(all_embeds_to_feed, return_logits=True)
                        else:
                            self.decoder.feed(all_embeds_to_feed)

                        # Schema tracking
                        for embed, is_last, token_id in feed_operations:
                            if token_id is not None:
                                self._current_unit_prefill_tokens.append(token_id)
                            else:
                                embed_dim = embed.shape[0] if len(embed.shape) > 1 else 1
                                self._current_unit_prefill_tokens.append(("img", embed_dim))
                    else:
                        # Sequential mode (original behavior)
                        for embed, is_last, token_id in feed_operations:
                            if mode == "VISION" and is_last:
                                self.pending_logits, _ = self.decoder.feed(embed, return_logits=True)
                            else:
                                self.decoder.feed(embed)

                            # Schema tracking
                            if token_id is not None:
                                self._current_unit_prefill_tokens.append(token_id)
                            else:
                                embed_dim = embed.shape[0] if len(embed.shape) > 1 else 1
                                self._current_unit_prefill_tokens.append(("img", embed_dim))

        # Process audio (aligned with original streaming_prefill)
        if has_audio:
            # Accumulate audio
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_waveform])

            # Calculate required length
            if self.audio_chunk_idx == 0:
                required_samples = int(self.FIRST_CHUNK_MS * self.SAMPLE_RATE / 1000)
                if len(self.audio_buffer) < required_samples:
                    padding = np.zeros(required_samples - len(self.audio_buffer), dtype=np.float32)
                    self.audio_buffer = np.concatenate([padding, self.audio_buffer])
            else:
                required_samples = int(self.CHUNK_MS * self.SAMPLE_RATE / 1000)

            # Check streaming chunk size from processor (aligned with original)
            need_samples = required_samples
            if self.processor is not None and hasattr(self.processor, "get_streaming_chunk_size"):
                need_samples = self.processor.get_streaming_chunk_size()

            if len(self.audio_buffer) < need_samples:
                return _make_result(False, f"audio not enough: need {need_samples} samples, only {len(self.audio_buffer)}")

            audio_chunk = self.audio_buffer[:need_samples]

            # Get audio features using processor
            if self.processor is not None and hasattr(self.processor, "process_audio_streaming"):
                batch_feature = self.processor.process_audio_streaming(
                    audio_chunk,
                    reset=False,
                    return_batch_feature=True,
                )

                if batch_feature is None or (hasattr(batch_feature, "audio_features") and batch_feature.audio_features.shape[-1] == 0):
                    return _make_result(False, "streaming audio processing returned empty")

                batch_feature.chunk_idx = self.audio_chunk_idx
                batch_feature.use_extra_context = True
                batch_feature.prefix_extra_frames = 0 if self.audio_chunk_idx == 0 else 2
                batch_feature.suffix_extra_frames = 2

                # Prepare audio data dict
                audio_data = {
                    "audio_features": batch_feature.audio_features,
                    "audio_feature_lens": batch_feature.audio_feature_lens,
                }

                # Get audio embedding using streaming method (aligned with original)
                audio_embeds_list = self.model.get_audio_embedding_streaming(
                    audio_data,
                    use_extra_context=batch_feature.use_extra_context,
                    prefix_extra_frames=batch_feature.prefix_extra_frames,
                    suffix_extra_frames=batch_feature.suffix_extra_frames,
                )

                # Flatten and concatenate embeddings (aligned with original)
                if audio_embeds_list and len(audio_embeds_list) > 0:
                    audio_embeds = torch.cat([t for g in audio_embeds_list for t in g], dim=0) if audio_embeds_list else None

                    if audio_embeds is not None:
                        self.pending_logits, _ = self.decoder.feed(audio_embeds, return_logits=True)

                        # Schema tracking
                        embed_dim = audio_embeds.shape[0] if len(audio_embeds.shape) > 1 else 1
                        self._current_unit_prefill_tokens.append(("audio", embed_dim))

            # Consume audio (aligned with original)
            if self.audio_chunk_idx == 0:
                if self.processor is not None and hasattr(self.processor, "_streaming_mel_processor"):
                    cfg = self.processor._streaming_mel_processor.get_config()
                    consumed_ms = int(cfg.get("effective_first_chunk_ms", self.FIRST_CHUNK_MS))
                    consumed_samples = int(consumed_ms * self.SAMPLE_RATE / 1000)
                else:
                    consumed_samples = int(self.FIRST_CHUNK_MS * self.SAMPLE_RATE / 1000)
            else:
                consumed_samples = int(self.CHUNK_MS * self.SAMPLE_RATE / 1000)

            self.audio_buffer = self.audio_buffer[consumed_samples:]
            self.audio_chunk_idx += 1

        # Process text
        if has_text:
            text_content = "".join(text_list) if isinstance(text_list, list) else str(text_list)
            text_token_ids = self.tokenizer.encode(text_content, add_special_tokens=False)

            if len(text_token_ids) > 0:
                text_embeds = self.decoder.embed_tokens(text_token_ids)

                if mode == "TEXT":
                    self.pending_logits, _ = self.decoder.feed(text_embeds, return_logits=True)
                else:
                    self.decoder.feed(text_embeds)

                for tid in text_token_ids:
                    self._current_unit_prefill_tokens.append(tid)

        self.current_mode = mode

        if mode == "VISION":
            self.audio_chunk_idx += 1

        # Save schema
        self.prefill_schema_tokens.append(self._current_unit_prefill_tokens)

        return _make_result(True)

    @torch.no_grad()
    def streaming_generate(
        self,
        prompt_wav_path: Optional[str] = None,
        max_new_speak_tokens_per_chunk: int = 20,
        decode_mode: str = "sampling",
        temperature: float = 0.7,
        top_k: int = 100,
        top_p: float = 0.8,
        listen_prob_scale: float = 1.0,
        listen_top_k: Optional[int] = None,
        text_repetition_penalty: float = 1.05,
        text_repetition_window_size: int = 512,
        # Simplex API parameters (dispatched to self.model.streaming_generate)
        session_id=None,
        **kwargs,
    ):
        """
        Generate response in streaming mode (aligned with original model).

        Supports two calling conventions:
        - **Duplex API**: streaming_generate(prompt_wav_path=..., decode_mode=...)
        - **Simplex API**: streaming_generate(session_id=..., generate_audio=...) — delegates to self.model

        Args:
            prompt_wav_path: Path to reference audio for TTS
            max_new_speak_tokens_per_chunk: Max tokens per chunk
            decode_mode: "sampling" or "greedy"
            temperature: Sampling temperature
            top_k, top_p: Sampling parameters
            listen_prob_scale: Scale for listen token probability
            listen_top_k: Force listen if in top-k
            text_repetition_penalty: Repetition penalty
            text_repetition_window_size: Window for repetition penalty
            session_id: (Simplex API) Session identifier

        Returns:
            Dict with generation results (duplex), or generator (simplex)
        """
        # Dispatch simplex API calls to the underlying OVMiniCPMO model
        if session_id is not None:
            return self.model.streaming_generate(
                session_id=session_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )

        start_time = time.time()

        if self.is_session_stop_set() or self.is_break_set():
            return {
                "is_listen": True,
                "text": "",
                "audio_waveform": self._generate_silence_waveform(),
                "end_of_turn": True,
                "current_time": self.audio_chunk_idx,
                "cost_llm": 0.0,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
                "cost_all": time.time() - start_time,
                "n_tokens": 0,
                "n_tts_tokens": 0,
            }

        # Check pending logits
        if self.pending_logits is None:
            return {
                "is_listen": True,
                "text": "",
                "audio_waveform": self._generate_silence_waveform(),
                "end_of_turn": False,
                "current_time": self.audio_chunk_idx,
                "cost_llm": 0.0,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
                "cost_all": time.time() - start_time,
                "n_tokens": 0,
                "n_tts_tokens": 0,
            }

        logits = self.pending_logits
        self.pending_logits = None

        # Force listen for initial chunks
        force_listen = self._streaming_generate_count < self.force_listen_count
        self._streaming_generate_count += 1

        total_hidden_in_unit = []
        total_ids_in_unit = []
        current_time = self.audio_chunk_idx
        is_listen = False
        end_of_turn = False

        llm_start_time = time.time()

        # Generate tokens (aligned with original streaming_generate)
        for j in range(max_new_speak_tokens_per_chunk):
            # ls_mode == "explicit": feed chunk_eos at max tokens
            if j == max_new_speak_tokens_per_chunk - 1:
                if self.ls_mode == "explicit":
                    self.decoder.feed(self.decoder.embed_token(self.chunk_eos_token_id))
                    self.total_ids.append(self.chunk_eos_token_id)
                    break

            if force_listen:
                last_id = torch.tensor([self.listen_token_id], dtype=torch.long)
            else:
                last_id = self.decoder.decode(
                    logits=logits,
                    mode=decode_mode,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    listen_top_k=listen_top_k,
                    listen_prob_scale=listen_prob_scale,
                    text_repetition_penalty=text_repetition_penalty,
                    text_repetition_window_size=text_repetition_window_size,
                )

                # Don't listen if turn not ended
                if last_id.item() == self.listen_token_id and not self.current_turn_ended:
                    last_id = torch.tensor([self.tts_bos_token_id], dtype=torch.long)

            self.total_ids.append(last_id.item())
            is_listen = last_id.item() == self.listen_token_id

            # Check termination
            if last_id.item() in self.chunk_terminator_token_ids:
                # ls_mode == "explicit": feed the terminator token back
                if self.ls_mode == "explicit":
                    logits, _ = self.decoder.feed(self.decoder.embed_token(last_id.item()), return_logits=True)
                break
            else:
                self.current_turn_ended = False

                if last_id.item() not in self.chunk_speak_token_ids:
                    self.res_ids.append(last_id.item())
                    self.speak_count += 1

                # Feed token and get next logits + hidden states
                logits, hidden = self.decoder.feed(self.decoder.embed_token(last_id.item()), return_logits=True)

                end_of_turn = last_id.item() in self.turn_terminator_token_ids
                if end_of_turn:
                    self.current_turn_ended = True

                if j != 0:
                    total_hidden_in_unit.append([last_id.item(), hidden, end_of_turn])
                    total_ids_in_unit.append(last_id.item())

        # Feed </unit> token
        self.decoder.feed(self.decoder.embed_token(self.unit_end_token_id))
        self.total_ids.append(self.unit_end_token_id)

        # Decode generated text
        generated_text = self.tokenizer.decode(total_ids_in_unit, skip_special_tokens=True) if total_ids_in_unit else ""

        # Register unit end
        input_type = self.current_mode.lower() if self.current_mode else "audio"
        self.decoder.register_unit_end(
            input_type=input_type,
            generated_tokens=total_ids_in_unit,
            is_listen=is_listen,
            generated_text=generated_text,
        )

        # Apply sliding window if configured
        cfg = self.decoder._window_config
        if cfg.sliding_window_mode == "context":
            self.decoder.enforce_window_with_context()
        elif cfg.sliding_window_mode == "basic":
            self.decoder.enforce_window()

        llm_end_time = time.time()

        if is_listen:
            self.total_hidden.append([])
            return {
                "is_listen": True,
                "text": "",
                "audio_waveform": self._generate_silence_waveform(),
                "end_of_turn": False,
                "current_time": current_time,
                "cost_llm": llm_end_time - llm_start_time,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
                "cost_all": time.time() - start_time,
                "n_tokens": len(total_ids_in_unit),
                "n_tts_tokens": 0,
            }

        self.total_hidden.append(total_hidden_in_unit)
        text = generated_text

        if not self.generate_audio:
            return {
                "is_listen": False,
                "text": text,
                "audio_waveform": None,
                "end_of_turn": end_of_turn,
                "current_time": current_time,
                "cost_llm": llm_end_time - llm_start_time,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
                "cost_all": time.time() - start_time,
                "n_tokens": len(total_ids_in_unit),
                "n_tts_tokens": 0,
            }

        # TTS generation (aligned with original model's pipeline)
        tts_start_time = time.time()
        tts_prep_start_time = time.time()
        tts_condition = self._convert_results_to_tts_input(total_hidden_in_unit)
        tts_prep_end_time = time.time()

        max_token_per_chunk = 25 + 1
        min_token_per_chunk = 25 + 1

        if end_of_turn:
            min_token_per_chunk = 0
        force_flush = False
        if self.tts_text_start_pos == 0:  # start of turn
            min_token_per_chunk = 0
            force_flush = True

        if self.tts_current_turn_start_time is None:
            self.tts_current_turn_start_time = current_time

        new_tokens = self._tts_generate_chunk(
            inputs_embeds=tts_condition,
            max_new_token=max_token_per_chunk,
            min_new_tokens=min_token_per_chunk,
            temperature=self.tts_temperature.item() if isinstance(self.tts_temperature, torch.Tensor) else float(self.tts_temperature),
            repetition_penalty=float(self.tts_repetition_penalty) if hasattr(self, "tts_repetition_penalty") else 1.0,
        )
        tts_end_time = time.time()

        # Update TTS state (aligned with original)
        # For OV stateful model, tts_past_key_values is a flag (True = state active, None = needs reset)
        if end_of_turn:
            self.tts_text_start_pos = 0
            self.tts_past_key_values = None  # Signal TTS state reset on next call
            self.tts_current_turn_start_time = None
        else:
            self.tts_past_key_values = True  # OV model manages KV cache internally
            self.tts_text_start_pos += tts_condition.shape[1] + (new_tokens.shape[1] if new_tokens is not None else 0)

        # Token2wav generation (must be before token2wav reset, aligned with original)
        token2wav_start_time = time.time()
        audio_waveform = None
        if self.model is not None and self.model.token2wav is not None:
            if new_tokens is not None:
                audio_waveform = self._generate_waveform_from_tokens(new_tokens, prompt_wav_path or self.prompt_wav_path, end_of_turn, force_flush=force_flush)
        token2wav_end_time = time.time()

        # Reset token2wav state after audio generation (aligned with original)
        # This ensures all tokens in buffer are processed before reset
        if end_of_turn:
            self._reset_token2wav_for_new_turn()

        if audio_waveform is None:
            audio_waveform = self._generate_silence_waveform()

        return {
            "is_listen": False,
            "text": text,
            "audio_waveform": audio_waveform,
            "end_of_turn": end_of_turn,
            "current_time": current_time,
            "cost_llm": llm_end_time - llm_start_time,
            "cost_tts_prep": tts_prep_end_time - tts_prep_start_time,
            "cost_tts": tts_end_time - tts_start_time,
            "cost_token2wav": token2wav_end_time - token2wav_start_time,
            "cost_all": time.time() - start_time,
            "n_tokens": len(total_ids_in_unit),
            "n_tts_tokens": new_tokens.numel() if new_tokens is not None else 0,
        }

    def _convert_results_to_tts_input(self, results):
        """Convert LLM hidden states to TTS input (aligned with original model).

        Uses config.audio_bos_token_id (from TTS config), NOT <|tts_bos|> token.

        Args:
            results: List of [token_id, hidden_state, end_of_turn] triples

        Returns:
            tts_embeds: [1, seq_len+1, hidden_size] - TTS conditioning input
        """
        # audio_bos_token_id comes from TTS config (aligned with original: self.model.tts.audio_bos_token_id)
        audio_bos_token_id = self._tts_audio_bos_token_id

        if len(results) == 0:
            # Only audio_bos token
            audio_bos = self.model.tts.embedding(torch.tensor([audio_bos_token_id], dtype=torch.long))
            return audio_bos.unsqueeze(0)

        llm_tokens = []
        llm_hidden = []
        for hidden in results:
            llm_tokens.append(hidden[0])  # token id
            llm_hidden.append(hidden[1].squeeze(0) if hidden[1] is not None else None)

        # Get TTS text embeddings for the tokens
        llm_tokens_tensor = torch.tensor(llm_tokens, dtype=torch.long)
        llm_embeds = self.model.tts.embedding(llm_tokens_tensor)

        # Project LLM hidden states through semantic projector + L2 normalize
        if all(h is not None for h in llm_hidden):
            llm_hidden_tensor = torch.cat(llm_hidden, dim=0)
            llm_hidden_tensor = self.model.tts.projector_semantic(llm_hidden_tensor)
            llm_hidden_tensor = torch.nn.functional.normalize(llm_hidden_tensor, p=2, dim=-1)

            # Combine text embedding + projected hidden (aligned with original)
            tts_embeds = llm_embeds + llm_hidden_tensor
        else:
            tts_embeds = llm_embeds

        # Append audio_bos token
        audio_bos = self.model.tts.embedding(torch.tensor([audio_bos_token_id], dtype=torch.long))
        tts_embeds = torch.cat([tts_embeds, audio_bos], dim=0)

        return tts_embeds.unsqueeze(0)

    def _tts_generate_chunk(
        self,
        inputs_embeds: torch.Tensor,
        max_new_token: int = 26,
        min_new_tokens: int = 26,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> Optional[torch.Tensor]:
        """Generate TTS audio codes from conditioning input (aligned with original).

        This implements the TTS autoregressive decoding loop using the
        OV TTS language model, code embedding, and code head.
        Matches original model's generate_chunk method.

        Args:
            inputs_embeds: TTS conditioning [1, seq_len, hidden_size]
            max_new_token: Maximum new tokens to generate
            min_new_tokens: Minimum new tokens before allowing EOS
            temperature: TTS sampling temperature
            repetition_penalty: Repetition penalty for TTS audio tokens

        Returns:
            Generated audio code tokens [1, num_tokens, num_vq] or None
        """
        num_audio_tokens = self._tts_num_audio_tokens
        num_vq = self._tts_num_vq
        eos_token = self._tts_eos_token

        # Prepare for first step
        if self.tts_past_key_values is None:
            self.model.tts.reset_state()

        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        past_len = self.tts_text_start_pos

        # Repetition penalty window size (aligned with original CustomRepetitionPenaltyLogitsProcessorRepeat)
        rep_penalty_window = 16

        # Pre-allocate output (aligned with original)
        new_tokens = torch.zeros(batch_size, max_new_token, num_vq, dtype=torch.long)

        last_t = 0
        for t in range(max_new_token):
            last_t = t

            if t == 0:
                # First step: feed conditioning input
                attention_mask = torch.ones((batch_size, past_len + seq_len), dtype=torch.long)
                position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long).unsqueeze(0)
                hidden_states = self.model.tts.forward(inputs_embeds, attention_mask, position_ids)
                current_pos = past_len + seq_len
            else:
                # Subsequent steps: feed code embedding of previous token
                code_embeds = self.model.tts.code_embedding(new_tokens[:, t - 1 : t, :])  # [batch, 1, hidden_size]
                attention_mask = torch.ones((batch_size, current_pos + 1), dtype=torch.long)
                position_ids = torch.tensor([[current_pos]], dtype=torch.long)
                hidden_states = self.model.tts.forward(code_embeds, attention_mask, position_ids)
                current_pos += 1

            # Get code prediction from last hidden state
            code_logits = self.model.tts.code_head(hidden_states[:, -1:, :])  # [1, 1, num_audio_tokens, num_vq]
            logits = code_logits[:, 0].float()  # [batch, num_audio_tokens, num_vq]

            # Reshape logits for sampling (aligned with original: permute then reshape)
            logits = logits.permute(0, 2, 1)  # [batch, num_vq, num_audio_tokens]
            logits = logits.reshape(-1, logits.size(2))  # [batch*num_vq, num_audio_tokens]

            # Apply temperature
            logits = logits / max(temperature, 1e-6)

            # Apply repetition penalty (aligned with original CustomRepetitionPenaltyLogitsProcessorRepeat)
            # This penalizes repeated audio codes within a sliding window
            if repetition_penalty != 1.0 and t > 0:
                # Get previous tokens: [batch, t, num_vq] -> permute -> [batch, num_vq, t] -> reshape -> [batch*num_vq, t]
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)
                logits_token = input_ids_sliced.reshape(input_ids_sliced.size(0) * input_ids_sliced.size(1), -1)
                # Apply sliding window
                if logits_token.size(1) > rep_penalty_window:
                    logits_token = logits_token[:, -rep_penalty_window:]
                # Compute frequency-based penalty (aligned with original)
                freq = F.one_hot(logits_token, logits.size(1)).sum(1)
                if freq.size(0) > num_audio_tokens:
                    freq[num_audio_tokens:].zero_()
                alpha = torch.pow(torch.tensor(repetition_penalty), freq.float())
                # Apply: divide positive logits, multiply negative logits
                logits_cont = logits.contiguous()
                inp = logits_cont * alpha
                oth = logits_cont / alpha
                logits = torch.where(logits_cont < 0, inp, oth)

            # Suppress EOS token before min_new_tokens
            if t < min_new_tokens:
                logits[:, eos_token] = -float("inf")

            # Sample
            scores = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(scores, num_samples=1)
            idx_next = idx_next.view(-1, num_vq)  # [batch, num_vq]

            # Check EOS
            finish = idx_next.eq(eos_token).any(1)
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            if finish.all():
                break

        # Return only generated tokens (exclude last incomplete/eos token)
        generated = new_tokens[:, 0:last_t, :]

        if generated.shape[1] == 0:
            return None

        return generated

    def _generate_waveform_from_tokens(
        self,
        new_tokens: torch.Tensor,
        prompt_wav_path: Optional[str],
        is_last_chunk: bool = False,
        force_flush: bool = False,
    ) -> Optional[np.ndarray]:
        """Convert TTS audio tokens to waveform (aligned with original model).

        Uses Token2wav.stream() with pre_lookahead for cross-chunk context,
        matching the original model's _generate_waveform_from_tokens exactly.

        Args:
            new_tokens: Generated audio code tokens [1, num_tokens, num_vq]
            prompt_wav_path: Path to prompt wav for voice cloning
            is_last_chunk: Whether this is the last chunk in the turn
            force_flush: Whether to force flush the buffer

        Returns:
            Audio waveform as numpy array, or None
        """
        if not self.token2wav_initialized:
            print("Warning: token2wav_initialized is uninitialized")
            return None

        if self.model.token2wav is None:
            return None

        CHUNK_SIZE = 25

        # Flatten tokens to list (aligned with original: reshape to (-1,))
        token_ids = torch.reshape(new_tokens, (-1,)).tolist()
        self.token2wav_buffer += token_ids

        # Check for chunk terminator tokens
        has_chunk_eos = any(tid in self.chunk_terminator_token_ids for tid in token_ids)

        pcm_bytes_list = []

        try:
            token2wav = self.model.token2wav

            # Process tokens with chunk buffering + pre_lookahead (aligned with original)
            if has_chunk_eos or force_flush:
                # When there is chunk_eos or force_flush, try to flush more content
                while len(self.token2wav_buffer) >= self.pre_lookahead + 5:
                    chunk_to_process = min(CHUNK_SIZE + self.pre_lookahead, len(self.token2wav_buffer))
                    pcm_bytes = token2wav.stream(
                        self.token2wav_buffer[:chunk_to_process],
                        prompt_wav=prompt_wav_path,
                    )
                    if pcm_bytes is not None:
                        pcm_bytes_list.append(pcm_bytes)
                    consumed = min(CHUNK_SIZE, chunk_to_process - self.pre_lookahead)
                    self.token2wav_buffer = self.token2wav_buffer[consumed:]
            else:
                while len(self.token2wav_buffer) >= CHUNK_SIZE + self.pre_lookahead:
                    pcm_bytes = token2wav.stream(
                        self.token2wav_buffer[: CHUNK_SIZE + self.pre_lookahead],
                        prompt_wav=prompt_wav_path,
                    )
                    if pcm_bytes is not None:
                        pcm_bytes_list.append(pcm_bytes)
                    self.token2wav_buffer = self.token2wav_buffer[CHUNK_SIZE:]

            # Flush remaining tokens on last chunk
            if is_last_chunk and len(self.token2wav_buffer) > 0:
                pcm_bytes = token2wav.stream(
                    self.token2wav_buffer,
                    prompt_wav=prompt_wav_path,
                    last_chunk=True,
                )
                if pcm_bytes is not None:
                    pcm_bytes_list.append(pcm_bytes)
                self.token2wav_buffer = []
        except Exception as e:
            print(f"Warning: Token2wav error: {e}")
            import traceback

            traceback.print_exc()
            return None

        if not pcm_bytes_list:
            return None

        # Merge PCM and convert to numpy array (24kHz, int16 -> float32)
        # Aligned with original model's PCM processing
        all_pcm = b"".join(pcm_bytes_list)
        if len(all_pcm) == 0:
            return None

        pcm_np = np.frombuffer(all_pcm, dtype="<i2")
        audio_waveform = pcm_np.astype(np.float32) / 32768.0

        # Left pad with zeros if audio is less than 1 second (24kHz), skip for last chunk
        # (aligned with original model)
        min_samples = 24000  # 1 second at 24kHz
        if not is_last_chunk and len(audio_waveform) < min_samples:
            pad_length = min_samples - len(audio_waveform)
            audio_waveform = np.pad(audio_waveform, (pad_length, 0), mode="constant", constant_values=0)

        return audio_waveform

    @staticmethod
    def _generate_silence_waveform(duration_sec: float = 1.0) -> np.ndarray:
        """Generate silence waveform (24kHz, aligned with original model)."""
        sample_rate = 24000
        num_samples = int(duration_sec * sample_rate)
        return np.zeros(num_samples, dtype=np.float32)

    def get_generated_text(self) -> str:
        """Get generated text from current session (aligned with original MiniCPMODuplex.get_generated_text)."""
        return self.tokenizer.decode(self.res_ids)

    def get_current_time(self) -> int:
        """Get current audio chunk index (aligned with original MiniCPMODuplex.get_current_time)."""
        return self.audio_chunk_idx

    def as_simplex(self, reset_session: bool = True, reset_token2wav_cache: bool = False) -> "OVMiniCPMO":
        """Convert this OVMiniCPMODuplex instance back to OVMiniCPMO for simplex mode.

        Aligned with original MiniCPMODuplex.as_simplex().

        Args:
            reset_session: If True, reset streaming session state.
            reset_token2wav_cache: If True, also reset token2wav cache.

        Returns:
            The underlying OVMiniCPMO model instance without reloading.
        """
        if reset_session:
            self.model.reset_session(reset_token2wav_cache=reset_token2wav_cache)
        return self.model


# ==================== OVToken2wav Inference Class ====================


class OVFlow:
    """
    OpenVINO-based Flow model for mel spectrogram generation.

    Uses two OpenVINO models:
    - flow_embeddings: Token embedding, encoder, and speaker projection
    - flow_estimator: DiT model for flow matching denoising
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "CPU",
        up_rate: int = 2,
        output_size: int = 80,
        n_timesteps: int = 10,
        inference_cfg_rate: float = 0.7,
    ):
        """
        Initialize OVFlow with OpenVINO models.

        Args:
            model_dir: Path to the directory containing OpenVINO flow models
            device: OpenVINO device (CPU, GPU, etc.)
            up_rate: Upsampling rate (token to mel ratio)
            output_size: Output mel dimension (default 80)
            n_timesteps: Number of diffusion steps (default 10)
            inference_cfg_rate: Classifier-free guidance rate
        """
        self.model_dir = Path(model_dir)
        self.ov_device = device

        # Flow parameters
        self.up_rate = up_rate
        self.output_size = output_size
        self.n_timesteps = n_timesteps
        self.inference_cfg_rate = inference_cfg_rate

        # Fixed shape configuration for GPU efficiency
        self.flow_emb_token_len = 0  # Set via set_fixed_shape()
        self.flow_emb_prompt_len = 0  # Set via set_fixed_shape()
        # Load OpenVINO flow embeddings model
        flow_emb_path = self.model_dir / FLOW_EMBEDDINGS_NAME
        print(f"⌛ Loading OpenVINO Flow embeddings model from {flow_emb_path}...")
        self.flow_embeddings = core.compile_model(str(flow_emb_path), "CPU")
        print(f"✅ Flow embeddings model loaded")

        # Load OpenVINO flow estimator model (DiT)
        flow_est_path = self.model_dir / FLOW_ESTIMATOR_NAME
        print(f"⌛ Loading OpenVINO Flow estimator model from {flow_est_path}...")
        self.flow_estimator = core.compile_model(str(flow_est_path), device)
        print(f"✅ Flow estimator model loaded")

        # Pre-generate random noise for deterministic inference
        self._init_rand_noise()

    def _init_rand_noise(self, max_len: int = 50 * 600):
        """Initialize random noise buffer for deterministic inference."""
        torch.manual_seed(0)
        self.rand_noise = torch.randn([1, self.output_size, max_len])

    def set_fixed_shapes(self, flow_emb_token_len: int = 0, flow_emb_prompt_len: int = 0):
        """Set fixed input shapes for flow_embeddings for GPU efficiency.

        When set, token and prompt_token inputs are padded to fixed lengths,
        and the model is reshaped accordingly. This avoids dynamic shape
        recompilation on GPU.

        Args:
            flow_emb_token_len: Fixed token length (0 = dynamic)
            flow_emb_prompt_len: Fixed prompt token length (0 = dynamic)
        """
        if flow_emb_token_len > 0 and flow_emb_prompt_len > 0:
            self.flow_emb_token_len = flow_emb_token_len
            self.flow_emb_prompt_len = flow_emb_prompt_len

            flow_emb_path = self.model_dir / FLOW_EMBEDDINGS_NAME
            model = core.read_model(str(flow_emb_path))
            # Reshape: token [1, token_len], token_len [1], prompt [1, prompt_len],
            #          prompt_len [1], embedding [1, 192]
            model.reshape(
                {
                    "token": [1, flow_emb_token_len],
                    "token_len": [1],
                    "prompt_token": [1, flow_emb_prompt_len],
                    "prompt_token_len": [1],
                    "embedding": [1, 192],
                }
            )
            self.flow_embeddings = core.compile_model(model, self.ov_device)
            print(f"  📐 Reshaped flow_embeddings: token=[1,{flow_emb_token_len}], prompt=[1,{flow_emb_prompt_len}]")
        else:
            self.flow_emb_token_len = 0
            self.flow_emb_prompt_len = 0

    def _run_flow_embeddings(self, token, token_len, prompt_token, prompt_token_len, embedding):
        """
        Run flow embeddings model to get encoder output and speaker embedding.

        When fixed shapes are set, inputs are padded and outputs are trimmed.

        Returns:
            h: Encoder output (batch, mel_len, output_size)
            spks: Projected speaker embedding (batch, output_size)
        """
        # Fixed shape mode: pad inputs to target length
        original_token_len = token.shape[1]
        original_prompt_len = prompt_token.shape[1]

        if self.flow_emb_token_len > 0:
            target_tok = self.flow_emb_token_len
            target_prompt = self.flow_emb_prompt_len

            # Pad token
            if original_token_len < target_tok:
                pad_len = target_tok - original_token_len
                token = torch.nn.functional.pad(token, (0, pad_len), value=0)
            else:
                token = token[:, :target_tok]

            # Pad prompt_token
            if original_prompt_len < target_prompt:
                pad_len = target_prompt - original_prompt_len
                prompt_token = torch.nn.functional.pad(prompt_token, (0, pad_len), value=0)
            else:
                prompt_token = prompt_token[:, :target_prompt]

        inputs = {
            "token": token,
            "token_len": token_len,  # Keep original lengths — model uses mask internally
            "prompt_token": prompt_token,
            "prompt_token_len": prompt_token_len,
            "embedding": embedding,
        }
        start_time = time.time()
        result = self.flow_embeddings(inputs)
        elapsed = time.time() - start_time
        h = torch.from_numpy(result[0].copy())
        spks = torch.from_numpy(result[1].copy())

        # Trim output if we padded: output mel_len = (prompt_len + token_len) * up_rate
        if self.flow_emb_token_len > 0:
            real_mel_len = (original_prompt_len + original_token_len) * self.up_rate
            h = h[:, :real_mel_len, :]

        return h, spks

    def _run_flow_estimator(self, x, mask, mu, t, spks, cond):
        """
        Run flow estimator (DiT) model for one denoising step.

        Returns:
            Estimated velocity field (batch, output_size, mel_len)
        """
        # OpenVINO directly accepts torch.Tensor
        inputs = {
            "x": x,
            "mask": mask,
            "mu": mu,
            "t": t,
            "spks": spks,
            "cond": cond,
        }
        result = self.flow_estimator(inputs)
        return torch.from_numpy(result[0].copy())

    def _solve_euler(self, z, t_span, mu, mask, spks, cond):
        """
        Euler ODE solver for flow matching with classifier-free guidance.
        """
        x = z
        t, dt = t_span[0], t_span[1] - t_span[0]

        # Prepare batched inputs for CFG (classifier-free guidance)
        batch_size = x.shape[0]
        mel_len = x.shape[2]
        dtype = spks.dtype
        device = x.device

        x_in = torch.zeros([2, self.output_size, mel_len], device=device, dtype=dtype)
        mask_in = torch.zeros([2, 1, mel_len], device=device, dtype=dtype)
        mu_in = torch.zeros([2, self.output_size, mel_len], device=device, dtype=dtype)
        t_in = torch.zeros([2], device=device, dtype=dtype)
        spks_in = torch.zeros([2, self.output_size], device=device, dtype=dtype)
        cond_in = torch.zeros([2, self.output_size, mel_len], device=device, dtype=dtype)

        for step in range(1, len(t_span)):
            # Fill in batched inputs for CFG
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            mu_in[1] = 0  # No condition for CFG
            t_in[:] = t
            spks_in[0] = spks
            spks_in[1] = 0  # No speaker for CFG
            cond_in[0] = cond
            cond_in[1] = 0  # No cond for CFG

            # Run estimator
            dphi_dt = self._run_flow_estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

            # Apply classifier-free guidance
            dphi_dt_cond, dphi_dt_uncond = dphi_dt[0:1], dphi_dt[1:2]
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt_cond - self.inference_cfg_rate * dphi_dt_uncond

            # Euler step
            x = x + dt * dphi_dt
            t = t + dt

            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x.float()

    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        n_timesteps: int = None,
    ):
        """
        Run flow inference using OpenVINO.

        Args:
            token: Speech tokens (batch, seq_len)
            token_len: Token lengths
            prompt_token: Prompt speech tokens
            prompt_token_len: Prompt token lengths
            prompt_feat: Prompt mel features (batch, mel_len, 80)
            prompt_feat_len: Prompt feature lengths
            embedding: Speaker embedding (batch, 192)
            n_timesteps: Number of diffusion steps (default: self.n_timesteps)

        Returns:
            mel: Generated mel spectrogram (batch, 80, mel_len)
        """
        if n_timesteps is None:
            n_timesteps = self.n_timesteps

        # Step 1: Run flow embeddings to get encoder output and projected speaker embedding
        h, spks = self._run_flow_embeddings(
            token.to(torch.int32), token_len.to(torch.int32), prompt_token.to(torch.int32), prompt_token_len.to(torch.int32), embedding.to(torch.float32)
        )

        # Step 2: Calculate mel lengths
        mel_len1 = prompt_feat.shape[1]  # prompt mel length
        mel_len2 = h.shape[1] - mel_len1  # generated mel length
        total_mel_len = mel_len1 + mel_len2

        # Step 3: Prepare conditions
        conds = torch.zeros([1, total_mel_len, self.output_size], device=h.device, dtype=h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)  # (batch, 80, mel_len)

        # Step 4: Prepare mask
        mask = torch.ones([1, 1, total_mel_len], dtype=h.dtype)

        # Step 5: Prepare mu (encoder output)
        mu = h.transpose(1, 2).contiguous()  # (batch, 80, mel_len)

        # Step 6: Speaker embedding is already projected (batch, 80)
        spks = spks.to(h.dtype)

        # Step 7: Initialize noise
        z = self.rand_noise[:, :, :total_mel_len].to(h.device).to(h.dtype)

        # Step 8: Create time span with cosine schedule
        t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=h.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)  # Cosine schedule

        # Step 9: Solve ODE with Euler method
        feat = self._solve_euler(z, t_span, mu, mask, spks, conds)

        # Step 10: Return only the generated part (exclude prompt)
        feat = feat[:, :, mel_len1:]

        return feat


class OVHiFT:
    """OpenVINO-based HiFT vocoder for waveform generation.

    Supports fixed input shape mode for GPU efficiency:
    When hift_input_len > 0, the model is reshaped to a fixed mel length.
    Input is padded to that length, and output is trimmed back to original size.
    This avoids dynamic shape recompilation on GPU.
    """

    def __init__(self, model_path: str, device: str = "CPU", hift_input_len: int = 0):
        """
        Initialize OVHiFT with OpenVINO model.

        Args:
            model_path: Path to the OpenVINO hift model directory or .xml file
            device: OpenVINO device (CPU, GPU, etc.)
            hift_input_len: Fixed mel input length. If > 0, model is reshaped
                           to [1, 80, hift_input_len] for GPU efficiency.
                           Input will be zero-padded and output trimmed.
        """
        if Path(model_path).is_dir():
            model_path = Path(model_path) / HIFT_NAME
        self.model_path = Path(model_path)
        self.ov_device = device
        self.hift_input_len = hift_input_len

        # Load OpenVINO model with optional reshape
        print(f"⌛ Loading OpenVINO HiFT model from {model_path}...")
        model = core.read_model(str(model_path))
        if self.hift_input_len > 0:
            model.reshape([1, 80, self.hift_input_len])
            print(f"  📐 Reshaped HiFT to fixed input: [1, 80, {self.hift_input_len}]")
        self.hift = core.compile_model(model, device)
        print(f"✅ HiFT model loaded on {device}")

        # ISTFT parameters (matching HiFTGenerator defaults)
        self.istft_params = {"n_fft": 16, "hop_len": 4}
        self.audio_limit = 0.99

        # HiFT upsamples mel by hop_size product: 480 samples per mel frame
        # (from conv_post upsampling chain)
        self.mel_to_samples_ratio = 480

        # Create hann window for istft
        from scipy.signal import get_window

        self.stft_window = torch.from_numpy(get_window("hann", self.istft_params["n_fft"], fftbins=True).astype(np.float32))

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Inverse STFT to convert spectral features to waveform."""
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(
            torch.complex(real, img),
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(magnitude.device),
        )
        return inverse_transform

    def inference(self, speech_feat: torch.Tensor) -> torch.Tensor:
        """
        Run HiFT inference using OpenVINO.

        Args:
            speech_feat: Mel spectrogram (batch, 80, mel_len)

        Returns:
            speech: Generated waveform (batch, samples)
        """
        # Convert to numpy for potential padding
        if isinstance(speech_feat, torch.Tensor):
            mel_input = speech_feat.cpu().numpy()
        else:
            mel_input = speech_feat

        # Fixed shape mode: pad input to target length, trim output after
        original_len = mel_input.shape[2]
        if self.hift_input_len > 0:
            target_len = self.hift_input_len
            if original_len < target_len:
                # Pad with zeros on the right
                pad_len = target_len - original_len
                mel_input = np.pad(mel_input, ((0, 0), (0, 0), (0, pad_len)), mode="constant", constant_values=0)
            else:
                # Truncate if longer than target (shouldn't normally happen)
                mel_input = mel_input[:, :, :target_len]
                original_len = target_len

        # Run OpenVINO inference - output is (batch, n_fft+2, time)
        start_time = time.time()
        result = self.hift(mel_input)
        elapsed = time.time() - start_time
        x = torch.from_numpy(result[0].copy())

        # Post-processing: exp, sin, istft, clamp
        n_fft = self.istft_params["n_fft"]
        magnitude = torch.exp(x[:, : n_fft // 2 + 1, :])
        phase = torch.sin(x[:, n_fft // 2 + 1 :, :])

        speech = self._istft(magnitude, phase)
        speech = torch.clamp(speech, -self.audio_limit, self.audio_limit)

        # Trim output to match original input length
        if self.hift_input_len > 0 and original_len < self.hift_input_len:
            original_samples = original_len * self.mel_to_samples_ratio
            speech = speech[:, :original_samples]

        return speech


class OVToken2wav:
    """
    OpenVINO-based Token2wav for converting speech tokens to waveform.

    This class combines:
    - OVFlow for mel spectrogram generation via flow matching
    - OVHiFT for waveform synthesis via neural vocoder
    - ONNX models for speech tokenization (s3tokenizer) and speaker embedding (campplus)
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "CPU",
        float16: bool = False,
        n_timesteps: int = 10,
        hift_input_len: int = 0,
        flow_emb_token_len: int = 0,
        flow_emb_prompt_len: int = 0,
    ):
        """
        Initialize OVToken2wav with OpenVINO models.

        Args:
            model_dir: Path to the directory containing OpenVINO models
            device: OpenVINO device (CPU, GPU, etc.)
            float16: Whether to use float16 (not used for OV, kept for API compatibility)
            n_timesteps: Number of diffusion steps for flow matching
            hift_input_len: Fixed mel length for HiFT (0 = dynamic).
                           GPU efficiency: reshape model to fixed shape to avoid recompilation.
            flow_emb_token_len: Fixed token length for flow_embeddings (0 = dynamic).
            flow_emb_prompt_len: Fixed prompt length for flow_embeddings (0 = dynamic).
        """
        self.model_dir = Path(model_dir)
        self.ov_device = device
        self.float16 = float16
        self.n_timesteps = n_timesteps

        # Load Flow model
        self.flow = OVFlow(
            model_dir=model_dir,
            device=device,
            up_rate=2,  # Default for stepaudio2
            output_size=80,
            n_timesteps=n_timesteps,
        )

        # Set fixed shapes for flow embeddings if specified
        if flow_emb_token_len > 0 and flow_emb_prompt_len > 0:
            self.flow.set_fixed_shapes(flow_emb_token_len, flow_emb_prompt_len)

        # Load HiFT vocoder (with optional fixed shape)
        self.hift = OVHiFT(model_path=model_dir, device=device, hift_input_len=hift_input_len)

        # Load s3tokenizer ONNX model
        s3tok_path = self.model_dir / "speech_tokenizer_v2_25hz.onnx"
        if s3tok_path.exists():
            import s3tokenizer

            print(f"⌛ Loading s3tokenizer model...")
            self.audio_tokenizer = s3tokenizer.load_model(str(s3tok_path)).cpu().eval()
            print(f"✅ s3tokenizer model loaded")
        else:
            self.audio_tokenizer = None
            print(f"⚠️ s3tokenizer model not found at {s3tok_path}")

        # Load campplus model for speaker embedding
        # Try OpenVINO IR first (.xml), then fall back to ONNX
        campplus_ir_path = self.model_dir / "campplus.xml"
        campplus_onnx_path = self.model_dir / "campplus.onnx"

        if campplus_ir_path.exists():
            print(f"⌛ Loading campplus model (OpenVINO IR)...")
            self.spk_model = core.compile_model(str(campplus_ir_path), device)
            print(f"✅ campplus model loaded (OpenVINO IR)")
        elif campplus_onnx_path.exists():
            print(f"⌛ Loading campplus model (ONNX via OpenVINO)...")
            self.spk_model = core.compile_model(str(campplus_onnx_path), device)
            print(f"✅ campplus model loaded (ONNX)")
        else:
            self.spk_model = None
            print(f"⚠️ campplus model not found at {campplus_ir_path} or {campplus_onnx_path}")

        self.cache = None

        # Stream config
        self.mel_cache_len = 8  # hard-coded, 160ms
        self.source_cache_len = int(self.mel_cache_len * 480)  # 50hz mel -> 24kHz wave
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len).astype(np.float32)).cpu()

        # HiFT cache
        self.hift_cache_dict = {}

    def _prepare_prompt(self, prompt_wav):
        """Prepare prompt data from audio file."""
        import s3tokenizer
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
        from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)  # [T]
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(mels.cpu(), mels_lens.cpu())

        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        # OpenVINO directly accepts torch.Tensor - no need for .cpu().numpy()
        spk_input = spk_feat.unsqueeze(dim=0)
        spk_result = self.spk_model(spk_input)
        spk_emb = torch.tensor(spk_result[0], device="cpu")

        audio, sample_rate = torchaudio.load(prompt_wav, backend="soundfile")
        audio = audio.mean(dim=0, keepdim=True)  # [1, T]
        if sample_rate != 24000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
        prompt_mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
        prompt_mels = prompt_mel.unsqueeze(0).cpu()
        prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device="cpu")
        prompt_mels = torch.nn.functional.pad(
            prompt_mels, (0, 0, 0, prompt_speech_tokens.shape[1] * self.flow.up_rate - prompt_mels.shape[1]), mode="replicate"
        )
        return prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens

    def __call__(self, generated_speech_tokens, prompt_wav):
        """
        Convert generated speech tokens to audio waveform.

        Args:
            generated_speech_tokens: List of generated speech token IDs
            prompt_wav: Path to prompt audio file

        Returns:
            wav bytes in WAV format
        """
        import io
        import torchaudio

        if self.cache is None:
            self.cache = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device="cpu")
        generated_speech_tokens_lens = torch.tensor([generated_speech_tokens.shape[1]], dtype=torch.int32, device="cpu")

        # Run flow inference
        mel = self.flow.inference(
            generated_speech_tokens,
            generated_speech_tokens_lens,
            prompt_speech_tokens,
            prompt_speech_tokens_lens,
            prompt_mels,
            prompt_mels_lens,
            spk_emb,
            self.n_timesteps,
        )

        # Run HiFT vocoder
        wav = self.hift.inference(speech_feat=mel)

        output = io.BytesIO()
        torchaudio.save(output, wav.cpu(), sample_rate=24000, format="wav")

        return output.getvalue()

    def reset_cache(self):
        """Reset prompt cache."""
        self.cache = None
        self.hift_cache_dict = {}
        self.stream_initialized = False

    def set_stream_cache(self, prompt_wav):
        """Initialize streaming cache (aligned with original Token2wav.set_stream_cache).

        Prepares prompt features and initializes HiFT cache for cross-chunk
        speech continuity. Must be called before stream().

        Args:
            prompt_wav: Path to prompt wav file

        Returns:
            Tuple of (None, hift_cache_dict) — flow cache is not used in OV version
        """
        self.cache = self._prepare_prompt(prompt_wav)
        self.hift_cache_dict = {
            "mel": torch.zeros(1, 80, 0),
            "source": torch.zeros(1, 1, 0),
            "speech": torch.zeros(1, 0),
        }
        self.stream_initialized = True
        return None, self.hift_cache_dict

    def stream(self, tokens, prompt_wav, last_chunk=False, return_waveform=False):
        """Streaming token2wav with HiFT mel/speech caching (aligned with original Token2wav.stream).

        Each call processes one chunk of speech tokens through flow → HiFT.
        Uses mel caching and speech overlap smoothing (hamming window) for
        cross-chunk audio continuity.

        Args:
            tokens: List of speech token IDs for current chunk
            prompt_wav: Path to prompt wav file
            last_chunk: Whether this is the last chunk
            return_waveform: If True, return float32 numpy array instead of int16 PCM bytes

        Returns:
            If return_waveform=True: float32 numpy array (aligned with original Token2wav.stream return_waveform=True)
            If return_waveform=False: int16 PCM bytes (default, matching original Token2wav.stream API)
        """
        import io

        if self.cache is None:
            self.cache = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache

        generated = torch.tensor([tokens], dtype=torch.int32, device="cpu")
        generated_lens = torch.tensor([generated.shape[1]], dtype=torch.int32, device="cpu")

        # Run flow inference for this chunk
        mel = self.flow.inference(
            generated, generated_lens, prompt_speech_tokens, prompt_speech_tokens_lens, prompt_mels, prompt_mels_lens, spk_emb, self.n_timesteps
        )

        # HiFT with mel cache for cross-chunk continuity
        hift_cache_mel = self.hift_cache_dict.get("mel", torch.zeros(1, 80, 0))
        hift_cache_speech = self.hift_cache_dict.get("speech", torch.zeros(1, 0))
        is_first_chunk = hift_cache_speech.shape[-1] == 0

        if hift_cache_mel.shape[2] > 0:
            mel_combined = torch.cat([hift_cache_mel, mel], dim=2)
        else:
            mel_combined = mel

        # Run HiFT inference
        speech = self.hift.inference(speech_feat=mel_combined)

        # Speech overlap smoothing with cached speech
        if not is_first_chunk and hift_cache_speech.shape[-1] > 0:
            # Fade overlap between cached tail and new speech head
            overlap_len = min(self.source_cache_len, speech.shape[-1], hift_cache_speech.shape[-1])
            if overlap_len > 0:
                fade_window = (
                    self.speech_window[: 2 * overlap_len]
                    if 2 * overlap_len <= self.speech_window.shape[0]
                    else torch.from_numpy(np.hamming(2 * overlap_len).astype(np.float32))
                )
                fade_in = fade_window[overlap_len:]  # second half: 0→1
                fade_out = fade_window[:overlap_len]  # first half: 1→0

                overlap_new = speech[:, :overlap_len]
                overlap_old = hift_cache_speech[:, -overlap_len:]

                blended = overlap_old * fade_out.unsqueeze(0) + overlap_new * fade_in.unsqueeze(0)
                speech = torch.cat([blended, speech[:, overlap_len:]], dim=1)

        # Update HiFT cache
        self.hift_cache_dict = {
            "mel": mel_combined[:, :, -self.mel_cache_len :].clone() if mel_combined.shape[2] >= self.mel_cache_len else mel_combined.clone(),
            "source": torch.zeros(1, 1, 0),
            "speech": speech[:, -self.source_cache_len :].clone() if speech.shape[-1] >= self.source_cache_len else speech.clone(),
        }

        # Trim output
        if is_first_chunk and not last_chunk:
            # First chunk: pad silence at start, trim tail for next overlap
            silence = torch.zeros(1, self.source_cache_len)
            if speech.shape[-1] > self.source_cache_len:
                speech = torch.cat([silence, speech[:, : -self.source_cache_len]], dim=1)
            else:
                speech = silence
        elif not last_chunk:
            # Middle chunks: trim tail for next overlap
            if speech.shape[-1] > self.source_cache_len:
                speech = speech[:, : -self.source_cache_len]
            else:
                speech = torch.zeros(1, 0)
        # Last chunk: output everything (no trimming)

        # Convert speech tensor to output format
        wav_np = speech.squeeze(0).cpu().numpy()
        wav_np = np.clip(wav_np, -1.0, 1.0)

        if return_waveform:
            # Return float32 numpy array with shape [1, samples] for torch.cat compatibility
            return wav_np.reshape(1, -1)

        # Convert to int16 PCM bytes (default, aligned with original Token2wav.stream)
        wav_int16 = (wav_np * 32767.0).astype("<i2")
        return wav_int16.tobytes()
