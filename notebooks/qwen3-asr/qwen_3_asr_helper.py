"""
Qwen3-ASR OpenVINO helper — fully self-contained.

Provides:
- convert_qwen3_asr_model()  — export PyTorch → OpenVINO IR (no optimum-intel dependency)
- OVQwen3ASRModel             — inference wrapper (same API as Qwen3ASRModel.transcribe)
"""

import gc
import json
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openvino as ov

# ── Optional imports ────────────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from qwen_asr.inference.utils import (
        SAMPLE_RATE,
        MAX_ASR_INPUT_SECONDS,
        SUPPORTED_LANGUAGES,
        AudioLike,
        AudioChunk,
        normalize_audios,
        normalize_language_name,
        validate_language,
        parse_asr_output,
        split_audio_into_chunks,
        merge_languages,
        chunk_list,
    )
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

    INFERENCE_UTILS_AVAILABLE = True
except ImportError:
    INFERENCE_UTILS_AVAILABLE = False
    SAMPLE_RATE = 16000
    MAX_ASR_INPUT_SECONDS = 1200
    SUPPORTED_LANGUAGES = ["Chinese", "English"]

try:
    import nncf
    NNCF_AVAILABLE = True
except ImportError:
    NNCF_AVAILABLE = False

core = ov.Core()


# ── Stateful helpers (adapted from optimum-intel) ──────────────────────────────

def _patch_stateful(ov_model: ov.Model):
    """
    Convert a decoder-only language model with explicit past_key_values I/O
    into a stateful model (KV-cache hidden as internal state variables).
    """
    from openvino import opset13
    from openvino._offline_transformations import apply_make_stateful_transformation

    kv_input_names = [
        name for inp in ov_model.inputs for name in inp.get_names() if "key_values" in name
    ]
    kv_output_names = [
        name for out in ov_model.outputs for name in out.get_names() if "present" in name
    ]
    not_kv_inputs = [
        inp for inp in ov_model.inputs
        if not any(n in kv_input_names for n in inp.get_names())
    ]
    if not kv_input_names or not kv_output_names:
        return

    # Fuse beam-search cache reorder (adds beam_idx parameter + Gather ops)
    main_input_name = "inputs_embeds"
    input_batch = ov_model.input(main_input_name).get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32,
                                 shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])

    batch_dim = 0
    for name in kv_input_names:
        port = ov_model.input(name)
        consumers = port.get_target_inputs()
        gather = opset13.gather(port, beam_idx, opset13.constant(batch_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()

    # Make stateful: hide KV I/O as internal ReadValue/Assign
    io_map = dict(zip(kv_input_names, kv_output_names))
    apply_make_stateful_transformation(ov_model, io_map)

    # Build state initializers from input shape
    input_ids = ov_model.input(main_input_name)
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]), opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [d.min_length for d in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [
                opset13.constant(np.array([d], dtype=np.int64)) if isinstance(d, int) else d
                for d in dims
            ]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(
                opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape
            )
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def _cleanup_torchscript_cache():
    """Remove cached torch.jit artifacts."""
    import torch
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


# ── Conversion ──────────────────────────────────────────────────────────────────

def convert_qwen3_asr_model(model_id: str, output_dir: str, quantization_config=None, **kwargs):
    """
    Convert Qwen3-ASR model to OpenVINO format (self-contained, no optimum-intel).

    Directly traces the 4 sub-models with ``ov.convert_model`` and applies
    stateful transformation to the language model for efficient KV-cache handling.

    Exported sub-models:
      1. openvino_audio_conv_model      — Conv2D audio frontend
      2. openvino_audio_encoder_model   — Transformer encoder layers
      3. openvino_text_embeddings_model — Token embedding layer
      4. openvino_language_model        — Decoder LLM (stateful, with int8 compression)

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen3-ASR-1.7B") or local path
        output_dir: Directory for exported OV models
        quantization_config: Optional dict for nncf.compress_weights (e.g. {"mode": "int4_sym_g128"})
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model conversion. Install: pip install torch")

    import torch
    import torch.nn as nn

    output_dir = Path(output_dir)
    expected_files = [
        "openvino_audio_conv_model.xml",
        "openvino_audio_encoder_model.xml",
        "openvino_text_embeddings_model.xml",
        "openvino_language_model.xml",
    ]
    if all((output_dir / f).exists() for f in expected_files):
        print(f"✅ {model_id} model already converted. Results in {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load PyTorch model ──────────────────────────────────────────────────
    print(f"⌛ {model_id} conversion started. Be patient, it may take some time.")
    print("⌛ Loading original model…")

    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

    config = Qwen3ASRConfig.from_pretrained(model_id)
    config.thinker_config.text_config._attn_implementation_autoset = False
    config.thinker_config.text_config._attn_implementation = "sdpa"

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id, config=config, torch_dtype=torch.float16,
    )
    model.eval()

    # Save processor & config
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        processor.save_pretrained(str(output_dir))
    except Exception as e:
        print(f"⚠️ Could not save processor: {e}")
    config.save_pretrained(str(output_dir))
    print("✅ Original model loaded")

    audio = model.thinker.audio_tower
    num_mel_bins = audio.config.num_mel_bins
    d_model = audio.config.d_model
    hidden_size = model.thinker.model.config.hidden_size

    # Helper: make fp16 model params traceable with fp32 inputs
    try:
        from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
    except ImportError:
        def __make_16bit_traceable(model):
            model.float()

    # ── 1. Text Embeddings ──────────────────────────────────────────────────
    emb_path = output_dir / "openvino_text_embeddings_model.xml"
    if not emb_path.exists():
        print("⌛ Converting text embeddings model…")
        embed_tokens = model.thinker.model.get_input_embeddings()
        __make_16bit_traceable(embed_tokens)
        ov_emb = ov.convert_model(
            embed_tokens,
            example_input=torch.ones([2, 2], dtype=torch.int64),
        )
        ov.save_model(ov_emb, emb_path)
        del ov_emb
        _cleanup_torchscript_cache()
        gc.collect()
        print("✅ Text embeddings model converted")

    # ── 2. Audio Conv (Conv2D frontend) ─────────────────────────────────────
    conv_path = output_dir / "openvino_audio_conv_model.xml"
    if not conv_path.exists():
        print("⌛ Converting audio conv model…")
        audio._orig_forward = audio.forward

        def forward_audio_conv(self, input_features):
            x = input_features.unsqueeze(1)
            x = nn.functional.gelu(self.conv2d1(x))
            x = nn.functional.gelu(self.conv2d2(x))
            x = nn.functional.gelu(self.conv2d3(x))
            b, c, f, t = x.size()
            x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
            x = self.conv_out(x)
            return x

        audio.forward = types.MethodType(forward_audio_conv, audio)
        __make_16bit_traceable(audio)
        ov_conv = ov.convert_model(
            audio,
            example_input={"input_features": torch.randn([3, num_mel_bins, 100], dtype=torch.float32)},
            input=[ov.PartialShape([-1, num_mel_bins, -1])],
        )
        ov.save_model(ov_conv, conv_path)
        del ov_conv
        audio.forward = audio._orig_forward
        _cleanup_torchscript_cache()
        gc.collect()
        print("✅ Audio conv model converted")

    # ── 3. Audio Encoder (Transformer layers) ───────────────────────────────
    enc_path = output_dir / "openvino_audio_encoder_model.xml"
    if not enc_path.exists():
        print("⌛ Converting audio encoder model…")
        audio._orig_forward = audio.forward
        orig_attn_impl = audio.config._attn_implementation
        # Force eager attention (flash/sdpa not exportable)
        audio.config._attn_implementation = "eager"
        for layer in audio.layers:
            layer.self_attn.config._attn_implementation = "eager"

        def forward_audio_encoder(self, hidden_states, cu_seqlens):
            for encoder_layer in self.layers:
                layer_outputs = encoder_layer(hidden_states, cu_seqlens)
                hidden_states = layer_outputs[0]
            hidden_states = self.ln_post(hidden_states)
            hidden_states = self.proj1(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.proj2(hidden_states)
            return hidden_states

        audio.forward = types.MethodType(forward_audio_encoder, audio)
        __make_16bit_traceable(audio)
        ov_enc = ov.convert_model(
            audio,
            example_input={
                "hidden_states": torch.randn([5, d_model], dtype=torch.float32),
                "cu_seqlens": torch.tensor([0, 5], dtype=torch.int32),
            },
            input=[ov.PartialShape([-1, d_model]), ov.PartialShape([-1])],
        )
        ov.save_model(ov_enc, enc_path)
        del ov_enc
        audio.forward = audio._orig_forward
        audio.config._attn_implementation = orig_attn_impl
        _cleanup_torchscript_cache()
        gc.collect()
        print("✅ Audio encoder model converted")

    # ── 4. Language Model (Decoder with KV-cache) ───────────────────────────
    lang_path = output_dir / "openvino_language_model.xml"
    if not lang_path.exists():
        print("⌛ Converting language model…")
        from transformers.cache_utils import DynamicCache

        # Patch DynamicLayer.lazy_initialization to create 4D empty tensors
        # instead of 1D torch.tensor([]), which fails in OV's aten::cat on dim=-2
        try:
            from transformers.cache_utils import DynamicLayer
            _orig_lazy_init = DynamicLayer.lazy_initialization

            def _ov_lazy_init(self, key_states):
                self.dtype, self.device = key_states.dtype, key_states.device
                shape = list(key_states.shape)
                shape[-2] = 0  # seq_len = 0
                self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
                self.values = torch.zeros(shape, dtype=self.dtype, device=self.device)
                self.is_initialized = True

            DynamicLayer.lazy_initialization = _ov_lazy_init
        except ImportError:
            _orig_lazy_init = None

        # Patch attention masking for traceable export (avoids SDPA vmap issues)
        _orig_mask_fns = {}
        try:
            from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

            def _ov_traceable_mask(batch_size, cache_position, kv_length, kv_offset=0,
                                   mask_function=None, attention_mask=None, **kwargs):
                """Causal + padding mask without vmap for OpenVINO tracing."""
                dtype = kwargs.get("dtype", torch.float32)
                q_length = cache_position.shape[0]
                device = cache_position.device
                # Causal mask via broadcasting (no vmap)
                q_idx = cache_position[:, None]                          # [q, 1]
                kv_idx = torch.arange(kv_length, device=device)[None, :] + kv_offset  # [1, kv]
                causal = kv_idx <= q_idx                                # [q, kv] bool
                # Apply padding mask
                if attention_mask is not None:
                    kv_end = kv_offset + kv_length
                    if attention_mask.shape[-1] >= kv_end:
                        pad = attention_mask[:, kv_offset:kv_end].bool()
                    else:
                        pad_need = kv_end - attention_mask.shape[-1]
                        pad = torch.cat([
                            attention_mask[:, kv_offset:].bool(),
                            torch.ones(batch_size, pad_need, dtype=torch.bool, device=device),
                        ], dim=1)
                    mask = causal[None, None, :, :] & pad[:, None, None, :]
                else:
                    mask = causal[None, None, :, :].expand(batch_size, 1, q_length, kv_length)
                min_val = torch.finfo(torch.float16).min
                return torch.where(
                    mask,
                    torch.tensor(0.0, device=device, dtype=dtype),
                    torch.tensor(min_val, device=device, dtype=dtype),
                )

            for attn_type in ("eager", "sdpa"):
                _orig_mask_fns[attn_type] = ALL_MASK_ATTENTION_FUNCTIONS.get(attn_type, None)
                ALL_MASK_ATTENTION_FUNCTIONS.register(attn_type, _ov_traceable_mask)
        except ImportError:
            pass

        lang_model = model.thinker

        def forward_wrap_thinker(
            self, input_ids=None, attention_mask=None, position_ids=None,
            past_key_values=None, inputs_embeds=None, use_cache=None,
            output_attentions=None, output_hidden_states=None,
            return_dict=None, cache_position=None,
        ):
            if past_key_values is not None:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True, return_dict=return_dict,
                cache_position=cache_position,
            )
            if past_key_values is not None:
                outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return (logits, outputs.past_key_values)

        lang_model._orig_forward = lang_model.forward
        lang_model.forward = types.MethodType(forward_wrap_thinker, lang_model)

        # Force cos/sin cache to fp32 for stable tracing
        if hasattr(lang_model, "model") and hasattr(lang_model.model, "layers"):
            for layer in lang_model.model.layers:
                if (hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb")
                        and hasattr(layer.self_attn.rotary_emb, "dtype")
                        and hasattr(layer.self_attn.rotary_emb, "inv_freq")
                        and hasattr(layer.self_attn.rotary_emb, "max_position_embeddings")
                        and hasattr(layer.self_attn.rotary_emb, "_set_cos_sin_cache")):
                    re = layer.self_attn.rotary_emb
                    if re.dtype != torch.float32:
                        re._set_cos_sin_cache(
                            seq_len=re.max_position_embeddings,
                            device=re.inv_freq.device,
                            dtype=torch.float32,
                        )

        __make_16bit_traceable(lang_model)

        num_layers = lang_model.model.config.num_hidden_layers
        num_kv_heads = lang_model.model.config.num_key_value_heads
        head_dim = lang_model.model.config.head_dim

        # Build example inputs with past_key_values
        pkv_shape = (2, num_kv_heads, 2, head_dim)
        input_embeds = torch.randn((2, 2, hidden_size))
        attention_mask = torch.ones([2, 4], dtype=torch.long)
        position_ids = torch.arange(2, 4).view(1, 1, -1).expand(3, 2, -1)
        past_key_values = [[torch.randn(pkv_shape) for _ in range(2)] for _ in range(num_layers)]

        example_input = {
            "inputs_embeds": input_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
        }

        # Build input/output names
        input_names = ["attention_mask", "position_ids"]
        output_names = ["logits"]
        for i in range(num_layers):
            input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        input_names.append("inputs_embeds")

        # Build PartialShapes: attention_mask, position_ids, PKVs, inputs_embeds
        input_shapes = [
            ov.PartialShape([-1, -1]),          # attention_mask
            ov.PartialShape([3, -1, -1]),        # position_ids
        ]
        input_shapes += [ov.PartialShape([-1, num_kv_heads, -1, head_dim])] * 2 * num_layers
        input_shapes += [ov.PartialShape([1, -1, hidden_size])]  # inputs_embeds

        ov_lang = ov.convert_model(lang_model, example_input=example_input, input=input_shapes)

        # Set tensor names
        for inp, name in zip(ov_lang.inputs, input_names):
            inp.get_tensor().set_names({name})
        for out, name in zip(ov_lang.outputs, output_names):
            out.get_tensor().set_names({name})

        # Apply stateful transformation (hides KV-cache as internal state)
        _patch_stateful(ov_lang)
        print("✅ Language model converted (stateful)")

        # Optional weight compression
        if quantization_config is not None and NNCF_AVAILABLE:
            print(f"⌛ Compressing weights ({quantization_config.get('mode', 'default')})…")
            ov_lang = nncf.compress_weights(ov_lang, **quantization_config)
            print("✅ Weight compression finished")
        elif quantization_config is None and NNCF_AVAILABLE:
            # Default: int8 asymmetric compression
            print("⌛ Applying default int8 weight compression…")
            ov_lang = nncf.compress_weights(ov_lang)
            print("✅ Weight compression finished")

        ov.save_model(ov_lang, lang_path)
        del ov_lang
        lang_model.forward = lang_model._orig_forward
        _cleanup_torchscript_cache()
        gc.collect()

        # Restore original attention mask functions
        try:
            from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask, eager_mask
            ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", sdpa_mask)
            ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask)
        except ImportError:
            pass

        # Restore original DynamicLayer.lazy_initialization
        if _orig_lazy_init is not None:
            from transformers.cache_utils import DynamicLayer
            DynamicLayer.lazy_initialization = _orig_lazy_init

    # ── Cleanup ─────────────────────────────────────────────────────────────
    del model
    gc.collect()
    print(f"✅ {model_id} conversion finished. Results in {output_dir}")


# ── Audio helpers ───────────────────────────────────────────────────────────────

def _get_feat_extract_output_lengths(input_lengths):
    """Output length after 3x stride-2 Conv2D layers."""
    input_lengths = np.asarray(input_lengths, dtype=np.int64)
    leave = input_lengths % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


class SinusoidsPositionEmbedding:
    """Sinusoidal positional embeddings — concatenated [sin | cos] layout."""

    def __init__(self, max_position_embeddings: int, embed_dim: int, max_timescale: float = 10000.0):
        half = embed_dim // 2
        log_inc = np.log(max_timescale) / (half - 1)
        inv = np.exp(-log_inc * np.arange(half, dtype=np.float32))
        t = np.arange(max_position_embeddings, dtype=np.float32)[:, None] * inv[None, :]
        self.positional_embedding = np.concatenate([np.sin(t), np.cos(t)], axis=1).astype(np.float32)

    def __getitem__(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


def load_audio_file(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio to float32 mono at target_sr."""
    audio_path = str(audio_path)
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")
    except Exception:
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
        except ImportError:
            import scipy.io.wavfile as wav
            sr, audio = wav.read(audio_path)
            audio = audio.astype(np.float32) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.asarray(audio, dtype=np.float32)


# ── Transcription result ────────────────────────────────────────────────────────

@dataclass
class ASRTranscription:
    language: str
    text: str
    time_stamps: Optional[Any] = None


# ── OpenVINO pipeline ───────────────────────────────────────────────────────────

class OVQwen3ASRPipeline:
    """Low-level OpenVINO inference pipeline for Qwen3-ASR (4 sub-models)."""

    def __init__(self, model_dir: str, device: str = "CPU"):
        self.model_dir = Path(model_dir)

        with open(self.model_dir / "config.json") as f:
            self.config = json.load(f)

        thinker = self.config.get("thinker_config", {})
        acfg = thinker.get("audio_config", {})
        tcfg = thinker.get("text_config", {})

        self.d_model = acfg["d_model"]
        self.num_mel_bins = acfg["num_mel_bins"]
        self.max_source_positions = acfg["max_source_positions"]
        self.n_window = acfg["n_window"]
        self.n_window_infer = acfg.get("n_window_infer", self.n_window * 2)
        self.hidden_size = tcfg["hidden_size"]
        self.model_max_text_tokens = int(tcfg.get("max_position_embeddings", 2048))
        self.audio_token_id = thinker["audio_token_id"]

        self.pos_emb = SinusoidsPositionEmbedding(self.max_source_positions, self.d_model)

        print(f"Loading OV models from {model_dir} on {device}…")
        self.audio_conv = core.compile_model(self.model_dir / "openvino_audio_conv_model.xml", device)
        self.audio_encoder = core.compile_model(self.model_dir / "openvino_audio_encoder_model.xml", device)
        self.text_emb = core.compile_model(self.model_dir / "openvino_text_embeddings_model.xml", device)

        lm = core.read_model(self.model_dir / "openvino_language_model.xml")
        self.lm_input_names = {k.get_any_name(): i for i, k in enumerate(lm.inputs)}

        # Detect whether language model expects 3-D position_ids [3, batch, seq]
        # (self-contained conversion) or 2-D [batch, seq] (optimum-cli).
        pos_shape = lm.input("position_ids").get_partial_shape()
        self._pos_ndim = len(pos_shape)

        compiled_lm = core.compile_model(lm, device)
        self.lm_req = compiled_lm.create_infer_request()
        print("All models loaded ✅")

    # ── audio tower ────────────────────────────────────────────────────────────
    def _audio_tower(self, input_features, feature_len):
        """Process single audio sample through conv + encoder."""
        chunk_size = self.n_window * 2
        aftercnn_len = int(_get_feat_extract_output_lengths(feature_len))
        chunk_num = int(np.ceil(feature_len / chunk_size))

        chunk_lengths = np.full(chunk_num, chunk_size, dtype=np.int64)
        rem = feature_len % chunk_size
        if rem > 0:
            chunk_lengths[-1] = rem

        feats_t = input_features.T
        chunks, start = [], 0
        for l in chunk_lengths:
            chunks.append(feats_t[start:start + int(l)])
            start += int(l)

        mx = max(c.shape[0] for c in chunks)
        padded = [np.pad(c, ((0, mx - c.shape[0]), (0, 0))) if c.shape[0] < mx else c for c in chunks]
        padded_feature = np.stack(padded).transpose(0, 2, 1).astype(np.float32)

        lens_cnn = _get_feat_extract_output_lengths(chunk_lengths)

        conv_out = self.audio_conv(padded_feature)[self.audio_conv.output(0)]
        conv_out = conv_out + self.pos_emb[conv_out.shape[1]][None, :, :]

        mask = np.zeros((len(chunk_lengths), conv_out.shape[1]), dtype=bool)
        for j, cl in enumerate(lens_cnn):
            mask[j, :int(cl)] = True

        hidden = conv_out[mask]

        win_cnn = mask.shape[-1] * (self.n_window_infer // chunk_size)
        cu = [0]
        cu += [win_cnn] * (aftercnn_len // win_cnn)
        r = aftercnn_len % win_cnn
        if r > 0:
            cu.append(r)
        cu_seqlens = np.cumsum(cu).astype(np.int32)

        # Process each segment independently (cu_seqlens attention isolation
        # is not correctly implemented in the exported model)
        segs = []
        for si in range(len(cu_seqlens) - 1):
            s, e = int(cu_seqlens[si]), int(cu_seqlens[si + 1])
            seg_h = hidden[s:e].astype(np.float32)
            seg_cu = np.array([0, e - s], dtype=np.int32)
            seg_out = self.audio_encoder({"hidden_states": seg_h, "cu_seqlens": seg_cu})[self.audio_encoder.output(0)]
            segs.append(seg_out)
        return np.concatenate(segs, axis=0)

    def _process_audio(self, input_features, feature_attention_mask):
        lens = np.sum(feature_attention_mask, axis=1).astype(np.int64)
        parts = []
        for i in range(input_features.shape[0]):
            fl = int(lens[i])
            parts.append(self._audio_tower(input_features[i, :, :fl], fl))
        return np.concatenate(parts, axis=0)

    # ── language model ─────────────────────────────────────────────────────────
    def _embed(self, ids):
        return self.text_emb(ids)[self.text_emb.output(0)]

    def _lm(self, embeds, attn, pos, last_only=True):
        # Expand 2-D position_ids → 3-D [3, batch, seq] when model requires it
        if self._pos_ndim == 3 and pos.ndim == 2:
            pos = np.stack([pos] * 3, axis=0)
        inp = {
            "inputs_embeds": embeds.astype(np.float32),
            "attention_mask": attn.astype(np.int64),
            "position_ids": pos.astype(np.int64),
        }
        if "beam_idx" in self.lm_input_names:
            inp["beam_idx"] = np.arange(embeds.shape[0], dtype=np.int32)
        self.lm_req.start_async(inp, share_inputs=False)
        self.lm_req.wait()
        logits = self.lm_req.get_tensor("logits").data
        return logits[:, -1:, :].copy() if last_only else logits.copy()

    _CHUNK = 256

    def _prefill(self, embeds, attn, pos):
        seq = embeds.shape[1]
        if seq <= self._CHUNK:
            return self._lm(embeds, attn, pos, last_only=True)
        logits = None
        for s in range(0, seq, self._CHUNK):
            e = min(s + self._CHUNK, seq)
            logits = self._lm(embeds[:, s:e, :], attn[:, :e], pos[:, s:e], last_only=True)
        return logits

    # ── transcribe ──────────────────────────────────────────────────────────────
    def transcribe_audio(self, audio: np.ndarray, processor, max_new_tokens: int = 512):
        """Transcribe a float32 mono 16 kHz numpy array."""
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio}]},
        ]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], audio=[audio], return_tensors="np", padding=True)

        input_ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        feats = inputs["input_features"]
        feat_mask = inputs["feature_attention_mask"]

        audio_features = self._process_audio(feats, feat_mask)
        embeds = self._embed(input_ids)

        amask = input_ids[0] == self.audio_token_id
        na, nf = int(amask.sum()), audio_features.shape[0]
        if na != nf:
            n = min(na, nf)
            positions = np.where(amask)[0][:n]
            embeds[0, positions] = audio_features[:n]
        else:
            embeds[0, amask] = audio_features

        pos = np.cumsum(attn, axis=-1) - 1
        pos = np.where(attn == 0, 0, pos)

        self.lm_req.reset_state()
        logits = self._prefill(embeds, attn, pos)

        tokenizer = processor.tokenizer
        eos = set()
        if tokenizer.eos_token_id is not None:
            eos.add(tokenizer.eos_token_id)
        for t in ["<|im_end|>", "<|endoftext|>"]:
            tid = tokenizer.convert_tokens_to_ids(t)
            if tid is not None and tid != tokenizer.unk_token_id:
                eos.add(tid)

        gen_ids = []
        cur_attn = attn.copy()
        for _ in range(max_new_tokens):
            tok = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            if tok in eos:
                break
            gen_ids.append(tok)
            ne = self._embed(np.array([[tok]]))
            cur_attn = np.concatenate([cur_attn, np.ones((1, 1), dtype=np.int64)], axis=1)
            np_pos = np.array([[cur_attn.shape[1] - 1]], dtype=np.int64)
            logits = self._lm(ne, cur_attn, np_pos)

        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)

        clean, lang = raw, "unknown"
        try:
            from qwen_asr.inference.utils import parse_asr_output
            lang, clean = parse_asr_output(raw, user_language=None)
        except Exception:
            if "<asr_text>" in clean:
                parts = clean.split("<asr_text>")
                if len(parts) > 1:
                    clean = parts[1].split("</asr_text>")[0] if "</asr_text>" in parts[1] else parts[1]
                    pfx = parts[0]
                    if "language" in pfx.lower():
                        lang = pfx.replace("language", "").strip()
            for sp in ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "</asr_text>"]:
                clean = clean.replace(sp, "")
            clean = clean.strip()

        return {"text": clean, "language": lang, "generated_tokens": len(gen_ids)}


# ── High-level model wrapper (same API as Qwen3ASRModel) ────────────────────────

class OVQwen3ASRModel:
    """
    OpenVINO Qwen3-ASR model with the same transcribe() API as qwen_asr.Qwen3ASRModel.
    """

    def __init__(self, model_dir: str, device: str = "CPU",
                 max_inference_batch_size: int = 32, max_new_tokens: int = 512):
        self.model_dir = Path(model_dir)
        self.device = device
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens

        self.pipeline = OVQwen3ASRPipeline(str(model_dir), device=device)

        # Load processor
        self.processor = None
        try:
            self.processor = Qwen3ASRProcessor.from_pretrained(str(model_dir))
            print("Processor loaded ✅")
        except Exception:
            try:
                # fallback: processor might be at HF hub
                with open(self.model_dir / "config.json") as f:
                    cfg = json.load(f)
                model_name = cfg.get("_name_or_path", "")
                if model_name:
                    self.processor = Qwen3ASRProcessor.from_pretrained(model_name)
                    print(f"Processor loaded from {model_name} ✅")
            except Exception as e:
                print(f"⚠️ Could not load processor: {e}")

        print("OVQwen3ASRModel ready ✅")

    @classmethod
    def from_pretrained(cls, model_dir: str, device: str = "CPU",
                        max_inference_batch_size: int = 32,
                        max_new_tokens: int = 512, **kwargs) -> "OVQwen3ASRModel":
        return cls(model_dir=model_dir, device=device,
                   max_inference_batch_size=max_inference_batch_size,
                   max_new_tokens=max_new_tokens)

    def get_support_languages(self) -> List[str]:
        return list(SUPPORTED_LANGUAGES)

    def get_supported_languages(self) -> List[str]:
        return self.get_support_languages()

    # ── core transcribe ─────────────────────────────────────────────────────────
    def transcribe(
        self,
        audio: Union[Any, List[Any]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[ASRTranscription]:
        """
        Transcribe audio(s).

        Args:
            audio: file path, URL, (np.ndarray, sr) tuple, or list of these
            context: ignored (kept for API compat)
            language: optional language hint
            return_time_stamps: not supported

        Returns:
            List[ASRTranscription]
        """
        if return_time_stamps:
            raise ValueError("return_time_stamps not supported in OpenVINO version")

        if self.processor is None:
            raise RuntimeError("Processor not loaded. Cannot transcribe.")

        # Normalize inputs
        if INFERENCE_UTILS_AVAILABLE:
            wavs = normalize_audios(audio)
        else:
            wavs = self._normalize_audio_simple(audio)

        n = len(wavs)

        # Normalize languages
        langs: List[Optional[str]]
        if language is None:
            langs = [None] * n
        else:
            langs = language if isinstance(language, list) else [language]
            if len(langs) == 1 and n > 1:
                langs = langs * n

        # Chunk long audio if needed
        max_chunk = MAX_ASR_INPUT_SECONDS if INFERENCE_UTILS_AVAILABLE else 600
        all_chunks = []
        for i, wav in enumerate(wavs):
            if INFERENCE_UTILS_AVAILABLE:
                parts = split_audio_into_chunks(wav, sr=SAMPLE_RATE, max_chunk_sec=max_chunk)
                for j, (cwav, off) in enumerate(parts):
                    all_chunks.append((i, cwav))
            else:
                all_chunks.append((i, wav))

        # Transcribe each chunk
        chunk_results: List[Tuple[int, str, str]] = []
        for orig_idx, cwav in all_chunks:
            res = self.pipeline.transcribe_audio(cwav, self.processor, max_new_tokens=self.max_new_tokens)
            chunk_results.append((orig_idx, res["language"], res["text"]))

        # Merge chunks per original audio
        out_langs: List[List[str]] = [[] for _ in range(n)]
        out_texts: List[List[str]] = [[] for _ in range(n)]
        for oi, lang_str, txt in chunk_results:
            out_langs[oi].append(lang_str)
            out_texts[oi].append(txt)

        results = []
        for i in range(n):
            merged_text = "".join([t for t in out_texts[i] if t])
            if INFERENCE_UTILS_AVAILABLE:
                merged_lang = merge_languages(out_langs[i])
            else:
                merged_lang = out_langs[i][0] if out_langs[i] else "unknown"
            results.append(ASRTranscription(language=merged_lang, text=merged_text))

        return results

    @staticmethod
    def _normalize_audio_simple(audio):
        """Fallback audio normalization when qwen_asr is not installed."""
        if isinstance(audio, list):
            return [OVQwen3ASRModel._to_wav(a) for a in audio]
        return [OVQwen3ASRModel._to_wav(audio)]

    @staticmethod
    def _to_wav(audio) -> np.ndarray:
        if isinstance(audio, str):
            return load_audio_file(audio)
        if isinstance(audio, tuple) and len(audio) == 2:
            wav, sr = audio
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            return wav.astype(np.float32)
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        raise ValueError(f"Unsupported audio type: {type(audio)}")
