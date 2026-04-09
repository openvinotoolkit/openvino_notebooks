"""
VoxCPM2 Text-to-Speech with OpenVINO — Conversion & Inference Helper

Converts VoxCPM2 model components to OpenVINO IR format and provides
an independent OpenVINO inference pipeline (no PyTorch at runtime).
"""

import gc
import sys
import math
import json
import types
from pathlib import Path
from typing import Optional, Tuple, List, Generator, Union

import numpy as np
import openvino as ov

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nncf

    NNCF_AVAILABLE = True
except ImportError:
    NNCF_AVAILABLE = False

try:
    from openvino import opset13
except ImportError:
    from openvino.runtime import opset13

# ---------------------------------------------------------------------------
# VoxCPM source on sys.path (needed for conversion)
# ---------------------------------------------------------------------------
VOXCPM_SRC = Path(__file__).parent / "VoxCPM" / "src"
if str(VOXCPM_SRC) not in sys.path:
    sys.path.insert(0, str(VOXCPM_SRC))

# ---------------------------------------------------------------------------
# OV model filenames
# ---------------------------------------------------------------------------
EMBED_TOKENS_NAME = "openvino_embed_tokens.xml"
FEAT_ENCODER_NAME = "openvino_feat_encoder.xml"
BASE_LM_NAME = "openvino_base_lm.xml"  # includes FSQ
RESIDUAL_LM_NAME = "openvino_residual_lm.xml"  # includes fusion_proj
DECODE_HEADS_NAME = "openvino_decode_heads.xml"  # dit_proj + stop_pred
DIT_ESTIMATOR_NAME = "openvino_dit_estimator.xml"
AUDIO_VAE_ENCODER_NAME = "openvino_audio_vae_encoder.xml"
AUDIO_VAE_DECODER_NAME = "openvino_audio_vae_decoder.xml"

core = ov.Core()


# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — Stateful-model utilities (adapted from Qwen3-TTS helper)
# ═══════════════════════════════════════════════════════════════════════════


def model_has_state(ov_model: ov.Model):
    return len(ov_model.get_sinks()) > 0


def fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, gather_dim):
    if any("beam_idx" in t.get_names() for t in ov_model.inputs):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    for input_name in key_value_input_names:
        port = ov_model.input(input_name)
        consumers = port.get_target_inputs()
        gather = opset13.gather(port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model, batch_dim):
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
            dims = [opset13.constant(np.array([d], dtype=np.int64)) if isinstance(d, int) else d for d in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(ov_model, not_kv_inputs, key_value_input_names, key_value_output_names, batch_dim, num_attention_heads):
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}
    for kv_in, kv_out in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_in] = kv_out
    apply_make_stateful_transformation(ov_model, input_output_map)
    build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model, num_main_outputs):
    """Make KV-cache inputs/outputs stateful.

    ``num_main_outputs`` = number of non-KV outputs (e.g. 1 for hidden_states).
    """
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs[2:-1]]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs[num_main_outputs:]]
    not_kv_inputs = [inp for inp in ov_model.inputs if not any(n in key_value_input_names for n in inp.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, gather_dim=0)
    make_stateful(ov_model, not_kv_inputs, key_value_input_names, key_value_output_names, batch_dim=0, num_attention_heads=1)


def cleanup_torchscript_cache():
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — PyTorch wrapper modules for OV conversion
# ═══════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    from openvino.frontend.pytorch.patch_model import __make_16bit_traceable

    # Inline RoPE helpers to avoid triggering voxcpm/__init__ (needs torchaudio)
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin):
        orig_dtype = q.dtype
        q, k = q.to(torch.float32), k.to(torch.float32)
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

    # ------------------------------------------------------------------
    def remove_weight_norm_all(module):
        """Remove weight_norm from all sub-modules, ignore errors."""
        for child in module.modules():
            try:
                nn.utils.remove_weight_norm(child)
            except (ValueError, AttributeError):
                pass

    # ------------------------------------------------------------------
    class FeatEncoderWrapper(nn.Module):
        """VoxCPMLocEnc + enc_to_lm_proj → single OV model."""

        def __init__(self, feat_encoder, enc_to_lm_proj):
            super().__init__()
            self.feat_encoder = feat_encoder
            self.enc_to_lm_proj = enc_to_lm_proj

        def forward(self, x):
            return self.enc_to_lm_proj(self.feat_encoder(x))

    # ------------------------------------------------------------------
    class MiniCPMLMWrapper(nn.Module):
        """Wraps MiniCPMModel with standard past_key_values I/O for OV.

        Supports optional merges:
        - ``fsq_layer``: appended after norm → outputs (raw, fsq, kv…)
        - ``fusion_proj``: prepended before LM layers → wider input
        """

        def __init__(self, model, fsq_layer=None, fusion_proj=None):
            super().__init__()
            self.layers = model.layers
            self.norm = model.norm
            self.rope_emb = model.rope_emb
            self.fsq = fsq_layer
            self.fusion_proj = fusion_proj
            cfg = model.config
            self._num_heads = cfg.num_attention_heads
            self._num_kv_heads = cfg.num_key_value_heads
            self._head_dim = cfg.kv_channels or (cfg.hidden_size // cfg.num_attention_heads)
            self._gqa_groups = self._num_heads // self._num_kv_heads

        def forward(self, attention_mask, position_ids, past_key_values, inputs_embeds):
            B, q_len, _ = inputs_embeds.shape

            # Optional fusion projection (residual_lm: [B,T,2H] → [B,T,H])
            if self.fusion_proj is not None:
                hidden = self.fusion_proj(inputs_embeds)
            else:
                hidden = inputs_embeds

            # RoPE
            if self.rope_emb is not None:
                pos_emb = self.rope_emb(position_ids[0])
            else:
                pos_emb = None

            # 4D causal mask  [B, 1, q_len, kv_total]
            kv_total = attention_mask.shape[-1]
            row_pos = position_ids.unsqueeze(-1)
            col_idx = torch.arange(kv_total, device=hidden.device).view(1, 1, -1)
            causal = (col_idx <= row_pos) & attention_mask.unsqueeze(1).bool()
            causal_mask = causal.unsqueeze(1).to(hidden.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(torch.float16).min

            present_flat = []
            for i, layer in enumerate(self.layers):
                pk, pv = past_key_values[i]
                hidden, new_k, new_v = self._layer_forward(layer, hidden, causal_mask, pos_emb, pk, pv)
                present_flat.extend([new_k, new_v])

            hidden = self.norm(hidden)

            # Optional FSQ (base_lm: return both raw and quantized)
            if self.fsq is not None:
                return (hidden, self.fsq(hidden)) + tuple(present_flat)
            return (hidden,) + tuple(present_flat)

        def _layer_forward(self, layer, hidden, mask, pos_emb, past_k, past_v):
            residual = hidden
            hidden = layer.input_layernorm(hidden)

            B, T, _ = hidden.shape
            attn = layer.self_attn

            q = attn.q_proj(hidden).view(B, T, self._num_heads, self._head_dim).transpose(1, 2)
            k = attn.k_proj(hidden).view(B, T, self._num_kv_heads, self._head_dim).transpose(1, 2)
            v = attn.v_proj(hidden).view(B, T, self._num_kv_heads, self._head_dim).transpose(1, 2)

            if pos_emb is not None:
                cos, sin = pos_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # concat KV cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

            # GQA expansion
            k_exp = k.unsqueeze(2).expand(-1, -1, self._gqa_groups, -1, -1)
            k_exp = k_exp.reshape(B, self._num_heads, -1, self._head_dim)
            v_exp = v.unsqueeze(2).expand(-1, -1, self._gqa_groups, -1, -1)
            v_exp = v_exp.reshape(B, self._num_heads, -1, self._head_dim)

            out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=mask)
            out = out.transpose(1, 2).reshape(B, T, self._num_heads * self._head_dim)
            hidden = attn.o_proj(out)

            hidden = residual + hidden

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

            return hidden, k, v

    # ------------------------------------------------------------------
    class DecodeHeadsWrapper(nn.Module):
        """dit_projections + stop_predictor → single model.

        Returns (dit_hidden, stop_logits) from (lm_hidden, res_hidden).
        """

        def __init__(self, lm_to_dit_proj, res_to_dit_proj, stop_proj, stop_head):
            super().__init__()
            self.lm_dit = lm_to_dit_proj
            self.res_dit = res_to_dit_proj
            self.stop_proj = stop_proj
            self.stop_act = nn.SiLU()
            self.stop_head = stop_head

        def forward(self, lm_hidden, res_hidden):
            dit_hidden = torch.cat([self.lm_dit(lm_hidden), self.res_dit(res_hidden)], dim=-1)
            stop_logits = self.stop_head(self.stop_act(self.stop_proj(lm_hidden)))
            return dit_hidden, stop_logits

    # ------------------------------------------------------------------
    class AudioVAEEncoderWrapper(nn.Module):
        """Wraps AudioVAE encoder → mu latent."""

        def __init__(self, encoder, fc_mu):
            super().__init__()
            self.encoder = encoder
            self.fc_mu = fc_mu

        def forward(self, x):
            h = self.encoder.block(x)
            return self.fc_mu(h)

    # ------------------------------------------------------------------
    class AudioVAEDecoderWrapper(nn.Module):
        """Wraps AudioVAE CausalDecoder with pre-computed sr_idx."""

        def __init__(self, decoder):
            super().__init__()
            self.model = decoder.model
            self.sr_cond_model = decoder.sr_cond_model

        def forward(self, z, sr_idx):
            x = z
            for layer, cond_layer in zip(self.model, self.sr_cond_model):
                if cond_layer is not None:
                    x = cond_layer(x, sr_idx)
                x = layer(x)
            return x


# ═══════════════════════════════════════════════════════════════════════════
# Part 3 — Conversion function
# ═══════════════════════════════════════════════════════════════════════════


def convert_voxcpm2_model(model_path: str, output_dir: str, quantization_config=None):
    """Convert all VoxCPM2 sub-models to OpenVINO IR.

    Args:
        model_path : path to model weights (VoxCPM2 directory with config.json,
                     model.safetensors, audiovae.pth / audiovae.safetensors).
        output_dir : directory where .xml/.bin files will be saved.
        quantization_config : optional nncf config dict for LM weight compression.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for model conversion")
    from openvino.frontend.pytorch.patch_model import __make_16bit_traceable

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already converted
    all_xml = [
        EMBED_TOKENS_NAME,
        FEAT_ENCODER_NAME,
        BASE_LM_NAME,
        RESIDUAL_LM_NAME,
        DECODE_HEADS_NAME,
        DIT_ESTIMATOR_NAME,
        AUDIO_VAE_ENCODER_NAME,
        AUDIO_VAE_DECODER_NAME,
    ]
    if all((output_dir / x).exists() for x in all_xml):
        print(f"✅ All models already converted in {output_dir}")
        return

    # ── Load original model ──────────────────────────────────────────
    print("⌛ Loading VoxCPM2 model …")
    # Mock torchaudio to avoid ImportError from VoxCPM v1 import chain
    if "torchaudio" not in sys.modules:
        import importlib.machinery

        _mock = types.ModuleType("torchaudio")
        _mock.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
        sys.modules["torchaudio"] = _mock
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    # Force CPU for conversion
    import voxcpm.model.voxcpm2 as _v2mod

    _orig_cuda_avail = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    model = VoxCPM2Model.from_local(str(model_path), optimize=False)
    torch.cuda.is_available = _orig_cuda_avail

    model = model.to("cpu").float().eval()
    print("✅ Model loaded")

    # Copy tokenizer + config
    import shutil

    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = model_path / fn
        if src.exists():
            shutil.copy2(src, output_dir / fn)

    # ── 1. embed_tokens ──────────────────────────────────────────────
    _convert_embed_tokens(model, output_dir)

    # ── 2. feat_encoder (+ enc_to_lm_proj) ───────────────────────────
    _convert_feat_encoder(model, output_dir)

    # ── 3. base_lm + fsq (stateful) ──────────────────────────────────
    _convert_base_lm(model, output_dir, quantization_config)

    # ── 4. residual_lm + fusion_proj (stateful) ──────────────────────
    _convert_residual_lm(model, output_dir, quantization_config)

    # ── 5. decode_heads (dit_proj + stop_pred) ────────────────────────
    _convert_decode_heads(model, output_dir)

    # ── 6. dit_estimator ──────────────────────────────────────────────
    _convert_dit_estimator(model, output_dir)

    # ── 7. audio_vae_encoder ──────────────────────────────────────────
    _convert_audio_vae_encoder(model, output_dir)

    # ── 8. audio_vae_decoder ──────────────────────────────────────────
    _convert_audio_vae_decoder(model, output_dir)

    del model
    gc.collect()
    print(f"\n✅ All models saved to {output_dir}")


# ── Individual conversion helpers ────────────────────────────────────────


def _convert_embed_tokens(model, output_dir):
    path = output_dir / EMBED_TOKENS_NAME
    if path.exists():
        print(f"  ✓ {EMBED_TOKENS_NAME} exists, skip")
        return
    print(f"  ⌛ Converting embed_tokens …")
    embed = model.base_lm.embed_tokens
    __make_16bit_traceable(embed)
    ov_model = ov.convert_model(embed, example_input=torch.ones([1, 4], dtype=torch.int64))
    ov_model.inputs[0].get_tensor().set_names({"input_ids"})
    ov_model.outputs[0].get_tensor().set_names({"embeddings"})
    ov.save_model(ov_model, path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"  ✅ embed_tokens done")


def _convert_feat_encoder(model, output_dir):
    path = output_dir / FEAT_ENCODER_NAME
    if path.exists():
        print(f"  ✓ {FEAT_ENCODER_NAME} exists, skip")
        return
    print(f"  ⌛ Converting feat_encoder …")
    wrapper = FeatEncoderWrapper(model.feat_encoder, model.enc_to_lm_proj)
    wrapper.eval()
    P = model.patch_size  # 4
    D = model.feat_dim  # 64
    example = torch.randn(1, 2, P, D)
    __make_16bit_traceable(wrapper)
    ov_model = ov.convert_model(
        wrapper,
        example_input=example,
        input=[ov.PartialShape([1, -1, P, D])],
    )
    ov_model.inputs[0].get_tensor().set_names({"audio_features"})
    ov_model.outputs[0].get_tensor().set_names({"encoded"})
    ov.save_model(ov_model, path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"  ✅ feat_encoder done")


def _build_lm_example(num_layers, hidden, kv_heads, head_dim, embed_dim=None):
    """Build example inputs + names + shapes for a stateful LM conversion."""
    if embed_dim is None:
        embed_dim = hidden
    pkv = [[torch.randn(1, kv_heads, 2, head_dim), torch.randn(1, kv_heads, 2, head_dim)] for _ in range(num_layers)]
    example = {
        "attention_mask": torch.ones([1, 4], dtype=torch.long),
        "position_ids": torch.arange(2, 4, dtype=torch.long).view(1, -1),
        "past_key_values": pkv,
        "inputs_embeds": torch.randn(1, 2, embed_dim),
    }
    in_names = ["attention_mask", "position_ids"]
    for i in range(num_layers):
        in_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
    in_names.append("inputs_embeds")
    in_shapes = [ov.PartialShape([-1, -1]), ov.PartialShape([-1, -1])]
    in_shapes += [ov.PartialShape([-1, kv_heads, -1, head_dim])] * (2 * num_layers)
    in_shapes += [ov.PartialShape([-1, -1, embed_dim])]
    return example, in_names, in_shapes


def _convert_base_lm(model, output_dir, quant_config):
    """Convert base_lm + fsq → single stateful model (2 main outputs)."""
    out_path = output_dir / BASE_LM_NAME
    if out_path.exists():
        print(f"  ✓ {BASE_LM_NAME} exists, skip")
        return
    print(f"  ⌛ Converting base_lm (+ fsq) …")

    wrapper = MiniCPMLMWrapper(model.base_lm, fsq_layer=model.fsq_layer)
    wrapper.eval()

    cfg = model.config.lm_config
    H = cfg.hidden_size
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.kv_channels or (H // cfg.num_attention_heads)
    num_layers = cfg.num_hidden_layers

    example, in_names, in_shapes = _build_lm_example(num_layers, H, kv_heads, head_dim)

    out_names = ["hidden_states", "fsq_hidden_states"]
    for i in range(num_layers):
        out_names.extend([f"present.{i}.key", f"present.{i}.value"])

    __make_16bit_traceable(wrapper)
    ov_model = ov.convert_model(wrapper, example_input=example, input=in_shapes)

    for inp, n in zip(ov_model.inputs, in_names):
        inp.get_tensor().set_names({n})
    for out, n in zip(ov_model.outputs, out_names):
        out.get_tensor().set_names({n})

    patch_stateful(ov_model, num_main_outputs=2)  # raw + fsq
    print(f"  ✅ base_lm (+ fsq) converted (stateful)")

    if quant_config is not None and NNCF_AVAILABLE:
        print(f"  ⌛ Compressing base_lm weights …")
        ov_model = nncf.compress_weights(ov_model, **quant_config)
        print(f"  ✅ base_lm weights compressed")

    ov.save_model(ov_model, out_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def _convert_residual_lm(model, output_dir, quant_config):
    """Convert fusion_proj + residual_lm → single stateful model."""
    out_path = output_dir / RESIDUAL_LM_NAME
    if out_path.exists():
        print(f"  ✓ {RESIDUAL_LM_NAME} exists, skip")
        return
    print(f"  ⌛ Converting residual_lm (+ fusion_proj) …")

    wrapper = MiniCPMLMWrapper(model.residual_lm, fusion_proj=model.fusion_concat_proj)
    wrapper.eval()

    cfg = model.config.lm_config
    H = cfg.hidden_size
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.kv_channels or (H // cfg.num_attention_heads)
    num_layers = model.config.residual_lm_num_layers

    # Input is concatenated [enc_out, feat_embed] → shape [B, T, 2*H]
    example, in_names, in_shapes = _build_lm_example(num_layers, H, kv_heads, head_dim, embed_dim=H * 2)

    out_names = ["hidden_states"]
    for i in range(num_layers):
        out_names.extend([f"present.{i}.key", f"present.{i}.value"])

    __make_16bit_traceable(wrapper)
    ov_model = ov.convert_model(wrapper, example_input=example, input=in_shapes)

    for inp, n in zip(ov_model.inputs, in_names):
        inp.get_tensor().set_names({n})
    for out, n in zip(ov_model.outputs, out_names):
        out.get_tensor().set_names({n})

    patch_stateful(ov_model, num_main_outputs=1)
    print(f"  ✅ residual_lm (+ fusion_proj) converted (stateful)")

    if quant_config is not None and NNCF_AVAILABLE:
        print(f"  ⌛ Compressing residual_lm weights …")
        ov_model = nncf.compress_weights(ov_model, **quant_config)
        print(f"  ✅ residual_lm weights compressed")

    ov.save_model(ov_model, out_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()


def _convert_decode_heads(model, output_dir):
    """Convert dit_projections + stop_predictor → single model."""
    path = output_dir / DECODE_HEADS_NAME
    if path.exists():
        print(f"  ✓ {DECODE_HEADS_NAME} exists, skip")
        return
    print(f"  ⌛ Converting decode_heads …")
    H = model.config.lm_config.hidden_size
    wrapper = DecodeHeadsWrapper(
        model.lm_to_dit_proj,
        model.res_to_dit_proj,
        model.stop_proj,
        model.stop_head,
    )
    wrapper.eval()
    __make_16bit_traceable(wrapper)
    ov_model = ov.convert_model(
        wrapper,
        example_input={"lm_hidden": torch.randn(1, H), "res_hidden": torch.randn(1, H)},
        input=[ov.PartialShape([1, H]), ov.PartialShape([1, H])],
    )
    ov_model.inputs[0].get_tensor().set_names({"lm_hidden"})
    ov_model.inputs[1].get_tensor().set_names({"res_hidden"})
    ov_model.outputs[0].get_tensor().set_names({"dit_hidden"})
    ov_model.outputs[1].get_tensor().set_names({"stop_logits"})
    ov.save_model(ov_model, path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"  ✅ decode_heads done")


def _convert_dit_estimator(model, output_dir):
    path = output_dir / DIT_ESTIMATOR_NAME
    if path.exists():
        print(f"  ✓ {DIT_ESTIMATOR_NAME} exists, skip")
        return
    print(f"  ⌛ Converting dit_estimator …")
    estimator = model.feat_decoder.estimator
    estimator.eval()
    D = model.feat_dim  # 64
    P = model.patch_size  # 4
    H_dit = model.config.dit_config.hidden_dim  # 1024
    H_lm = model.config.lm_config.hidden_size  # 2048
    # The mu input is [B, H_lm] (lm_to_dit_proj output1024 + res_to_dit_proj output1024 = 2048)
    # which gets reshaped inside the DiT as [B, H_lm//H_dit, H_dit] = [B, 2, 1024]
    __make_16bit_traceable(estimator)
    example = {
        "x": torch.randn(2, D, P),
        "mu": torch.randn(2, H_lm),
        "t": torch.randn(2),
        "cond": torch.randn(2, D, P),
        "dt": torch.zeros(2),
    }
    ov_model = ov.convert_model(
        estimator,
        example_input=example,
        input=[
            ov.PartialShape([-1, D, P]),
            ov.PartialShape([-1, H_lm]),
            ov.PartialShape([-1]),
            ov.PartialShape([-1, D, -1]),
            ov.PartialShape([-1]),
        ],
    )
    for inp, name in zip(ov_model.inputs, ["x", "mu", "t", "cond", "dt"]):
        inp.get_tensor().set_names({name})
    ov_model.outputs[0].get_tensor().set_names({"output"})
    ov.save_model(ov_model, path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"  ✅ dit_estimator done")


def _convert_audio_vae_encoder(model, output_dir):
    path = output_dir / AUDIO_VAE_ENCODER_NAME
    if path.exists():
        print(f"  ✓ {AUDIO_VAE_ENCODER_NAME} exists, skip")
        return
    print(f"  ⌛ Converting audio_vae_encoder …")
    vae = model.audio_vae.float()
    remove_weight_norm_all(vae.encoder)
    wrapper = AudioVAEEncoderWrapper(vae.encoder, vae.encoder.fc_mu)
    wrapper.eval()
    # Encoder runs at float32
    chunk = model.chunk_size  # prod(encoder_rates) = 640
    example = torch.randn(1, 1, chunk * 4)
    ov_model = ov.convert_model(wrapper, example_input=example, input=[ov.PartialShape([1, 1, -1])])
    ov_model.inputs[0].get_tensor().set_names({"audio"})
    ov_model.outputs[0].get_tensor().set_names({"latent"})
    ov.save_model(ov_model, path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"  ✅ audio_vae_encoder done")


def _convert_audio_vae_decoder(model, output_dir):
    path = output_dir / AUDIO_VAE_DECODER_NAME
    if path.exists():
        print(f"  ✓ {AUDIO_VAE_DECODER_NAME} exists, skip")
        return
    print(f"  ⌛ Converting audio_vae_decoder …")
    vae = model.audio_vae.float()
    remove_weight_norm_all(vae.decoder)
    wrapper = AudioVAEDecoderWrapper(vae.decoder)
    wrapper.eval()

    D = model.feat_dim  # 64
    P = model.patch_size  # 4
    # sr_idx for 48kHz with boundaries [20000, 30000, 40000] → bucket 3
    sr_boundaries = model.audio_vae.sr_bin_boundaries
    out_sr = model.audio_vae.out_sample_rate
    sr_idx = torch.bucketize(
        torch.tensor([out_sr], dtype=torch.int32),
        torch.tensor(sr_boundaries, dtype=torch.int32),
    )

    example = (torch.randn(1, D, P * 2), sr_idx)
    ov_model = ov.convert_model(
        wrapper,
        example_input=example,
        input=[ov.PartialShape([1, D, -1]), ov.PartialShape([1])],
    )
    ov_model.inputs[0].get_tensor().set_names({"latent"})
    ov_model.inputs[1].get_tensor().set_names({"sr_idx"})
    ov_model.outputs[0].get_tensor().set_names({"audio"})
    ov.save_model(ov_model, path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"  ✅ audio_vae_decoder done")


# ═══════════════════════════════════════════════════════════════════════════
# Part 4 — OpenVINO inference pipeline (no PyTorch dependency)
# ═══════════════════════════════════════════════════════════════════════════


class OVVoxCPM2Model:
    """Pure-OpenVINO inference pipeline for VoxCPM2 TTS.

    Independent of the original PyTorch model — only depends on converted
    OpenVINO IR files, tokenizer, and config.json.
    """

    def __init__(self, model_dir: str, device: str = "CPU"):
        self.model_dir = Path(model_dir)
        self.device = device

        # Load config
        with open(self.model_dir / "config.json") as f:
            self.config = json.load(f)

        lm_cfg = self.config["lm_config"]
        self.hidden_size = lm_cfg["hidden_size"]
        self.patch_size = self.config["patch_size"]
        self.feat_dim = self.config["feat_dim"]
        self.vocab_size = lm_cfg["vocab_size"]

        vae_cfg = self.config.get("audio_vae_config", {})
        self.encode_sample_rate = vae_cfg.get("sample_rate", 16000)
        self.out_sample_rate = vae_cfg.get("out_sample_rate", 48000)
        self.sample_rate = self.out_sample_rate
        encoder_rates = vae_cfg.get("encoder_rates", [2, 5, 8, 8])
        decoder_rates = vae_cfg.get("decoder_rates", [8, 6, 5, 2, 2, 2])
        self.chunk_size = math.prod(encoder_rates)  # 640
        self.decode_chunk_size = math.prod(decoder_rates)  # 960
        sr_boundaries = vae_cfg.get("sr_bin_boundaries", [20000, 30000, 40000])
        # Pre-compute sr_idx for the output sample rate
        self._sr_idx = np.searchsorted(sr_boundaries, self.out_sample_rate).astype(np.int32).reshape(1)

        # Special tokens
        self.audio_start_token = 101
        self.audio_end_token = 102
        self.ref_audio_start_token = 103
        self.ref_audio_end_token = 104

        # Load tokenizer
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        # Pre-compute multi-character Chinese token set for splitting
        vocab = self._tokenizer.get_vocab()
        self._multichar_chinese = {tok for tok in vocab if len(tok) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in tok)}

        # Compile OV models
        self._load_models()

    def _load_models(self):
        d = self.device
        md = self.model_dir

        self._embed_tokens = core.compile_model(md / EMBED_TOKENS_NAME, d)
        self._feat_encoder = core.compile_model(md / FEAT_ENCODER_NAME, d)
        self._decode_heads = core.compile_model(md / DECODE_HEADS_NAME, d)
        self._dit_est = core.compile_model(md / DIT_ESTIMATOR_NAME, d)
        self._vae_enc = core.compile_model(md / AUDIO_VAE_ENCODER_NAME, d)
        self._vae_dec = core.compile_model(md / AUDIO_VAE_DECODER_NAME, d)

        # Stateful LMs — need InferRequest for KV-cache management
        self._base_lm_compiled = core.compile_model(md / BASE_LM_NAME, d)
        self._base_lm = self._base_lm_compiled.create_infer_request()
        self._base_lm_n_out = len(self._base_lm_compiled.outputs)
        self._residual_lm_compiled = core.compile_model(md / RESIDUAL_LM_NAME, d)
        self._residual_lm = self._residual_lm_compiled.create_infer_request()
        self._residual_lm_n_out = len(self._residual_lm_compiled.outputs)

    def _infer_stateful(self, request, inputs_embeds, attention_mask, position_ids, n_out=1):
        """Run inference on a stateful LM model, returns list of output numpy arrays."""
        request.infer(
            {
                "inputs_embeds": inputs_embeds.astype(np.float32),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "beam_idx": np.array([0], dtype=np.int32),
            }
        )
        return [request.get_output_tensor(i).data.copy() for i in range(n_out)]

    # ── Tokenizer ─────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> np.ndarray:
        # No special tokens (original model doesn't prepend BOS)
        enc = self._tokenizer.encode(text, add_special_tokens=False)
        tokens = enc.tokens
        # Split multi-character Chinese tokens into single characters
        processed = []
        for tok in tokens:
            clean = tok.replace("\u2581", "")  # remove SentencePiece ▁ prefix
            if clean in self._multichar_chinese:
                processed.extend(list(clean))
            else:
                processed.append(tok)
        ids = [self._tokenizer.token_to_id(t) for t in processed]
        return np.array(ids, dtype=np.int64)

    # ── Audio encode / decode ─────────────────────────────────────────

    def encode_wav(self, wav_path: str, padding_mode: str = "right") -> np.ndarray:
        """Load & VAE-encode an audio file → latent patches [T, P, D]."""
        import librosa

        audio, _ = librosa.load(wav_path, sr=self.encode_sample_rate, mono=True)
        audio = audio.astype(np.float32)

        patch_len = self.patch_size * self.chunk_size
        remainder = len(audio) % patch_len
        if remainder != 0:
            pad_size = patch_len - remainder
            if padding_mode == "left":
                audio = np.pad(audio, (pad_size, 0))
            else:
                audio = np.pad(audio, (0, pad_size))

        audio_input = audio.reshape(1, 1, -1)
        latent = self._vae_enc({"audio": audio_input})[0]  # [1, D, T_latent]
        # reshape to patches: [T_patch, P, D]
        D = self.feat_dim
        P = self.patch_size
        latent = latent[0]  # [D, T_latent]
        T_latent = latent.shape[1]
        T_patch = T_latent // P
        # [D, T_latent] → [D, T_patch, P] → [T_patch, P, D]
        latent = latent[:, : T_patch * P].reshape(D, T_patch, P).transpose(1, 2, 0)
        return latent  # [T_patch, P, D]

    def decode_latent(self, latent: np.ndarray) -> np.ndarray:
        """Decode VAE latent [1, D, T] → audio waveform [1, 1, audio_len]."""
        audio = self._vae_dec({"latent": latent.astype(np.float32), "sr_idx": self._sr_idx})[0]
        return audio

    # ── CFM Euler solver ──────────────────────────────────────────────

    def _cfm_solve(self, mu: np.ndarray, cond: np.ndarray, n_timesteps: int, cfg_value: float) -> np.ndarray:
        """Run the Euler ODE solver with CFG. All numpy.

        mu   : [B, H_lm]   H_lm = 2 * H_dit
        cond : [B, D, P]
        Returns: [B, D, P]
        """
        B = mu.shape[0]
        D = self.feat_dim
        P = self.patch_size
        z = np.random.randn(B, D, P).astype(np.float32)

        t_span = np.linspace(1.0, 0.0, n_timesteps + 1, dtype=np.float32)
        sway = 1.0
        t_span = t_span + sway * (np.cos(np.pi / 2 * t_span) - 1.0 + t_span)

        x = z.copy()
        t = float(t_span[0])
        dt = float(t_span[0] - t_span[1])
        zero_init_steps = max(1, int(len(t_span) * 0.04))

        for step in range(1, len(t_span)):
            if step <= zero_init_steps:
                dphi = np.zeros_like(x)
            else:
                # Double-batch for CFG (positive + negative)
                x_in = np.concatenate([x, x], axis=0).astype(np.float32)
                mu_in = np.zeros([2 * B, mu.shape[1]], dtype=np.float32)
                mu_in[:B] = mu
                t_in = np.full(2 * B, t, dtype=np.float32)
                dt_in = np.zeros(2 * B, dtype=np.float32)  # mean_mode=false
                cond_in = np.concatenate([cond, cond], axis=0).astype(np.float32)

                out = self._dit_est({"x": x_in, "mu": mu_in, "t": t_in, "cond": cond_in, "dt": dt_in})[0]
                dphi_pos = out[:B]
                dphi_neg = out[B:]

                # CFG-zero-star
                pos_flat = dphi_pos.reshape(B, -1)
                neg_flat = dphi_neg.reshape(B, -1)
                dot = np.sum(pos_flat * neg_flat, axis=1, keepdims=True)
                sq = np.sum(neg_flat**2, axis=1, keepdims=True) + 1e-8
                st = (dot / sq).reshape(B, 1, 1)

                dphi = dphi_neg * st + cfg_value * (dphi_pos - dphi_neg * st)

            x = x - dt * dphi
            t = t - dt
            if step < len(t_span) - 1:
                dt = float(t - t_span[step + 1])

        return x

    # ── Make ref prefix ───────────────────────────────────────────────

    def _make_ref_prefix(self, ref_feat: np.ndarray):
        """Build [ref_start, ref_audio, ref_end] prefix arrays.

        ref_feat : [T_ref, P, D]
        Returns  : tokens, feats, text_mask, audio_mask  (1-D / 2-D numpy)
        """
        T = ref_feat.shape[0]
        P, D = self.patch_size, self.feat_dim
        z1 = np.zeros((1, P, D), dtype=np.float32)

        tokens = np.concatenate(
            [
                np.array([self.ref_audio_start_token], dtype=np.int64),
                np.zeros(T, dtype=np.int64),
                np.array([self.ref_audio_end_token], dtype=np.int64),
            ]
        )
        feats = np.concatenate([z1, ref_feat, z1], axis=0)
        t_mask = np.concatenate([np.ones(1, dtype=np.int32), np.zeros(T, dtype=np.int32), np.ones(1, dtype=np.int32)])
        a_mask = np.concatenate([np.zeros(1, dtype=np.int32), np.ones(T, dtype=np.int32), np.zeros(1, dtype=np.int32)])
        return tokens, feats, t_mask, a_mask

    # ── Core autoregressive inference ─────────────────────────────────

    def _inference(
        self,
        text_token: np.ndarray,
        text_mask: np.ndarray,
        audio_feat: np.ndarray,
        audio_mask: np.ndarray,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
    ):
        """Core autoregressive generation loop.

        All inputs shape [1, seq_len, …].
        When streaming=False, returns (latent_pred, pred_feats, context_len).
        When streaming=True, yields (latent_chunk, pred_feats, context_len) per step.
        """
        _, T, P, D = audio_feat.shape

        # 1. Encode audio features
        feat_embed = self._feat_encoder({"audio_features": audio_feat})[0]  # [1, T, H]

        # 2. Text embeddings (scale_emb=1.0 since use_mup=false)
        text_embed = self._embed_tokens({"input_ids": text_token})[0]  # [1, T, H]

        # 3. Combined embedding
        tm = text_mask[..., np.newaxis].astype(np.float32)
        am = audio_mask[..., np.newaxis].astype(np.float32)
        combined = tm * text_embed + am * feat_embed

        # 4. Base LM prefill (includes FSQ — returns [raw, fsq])
        self._base_lm.reset_state()
        pos_ids = np.arange(T, dtype=np.int64).reshape(1, -1)
        attn_mask = np.ones([1, T], dtype=np.int64)
        base_outs = self._infer_stateful(
            self._base_lm,
            combined,
            attn_mask,
            pos_ids,
            n_out=self._base_lm_n_out,
        )
        raw_out, fsq_out = base_outs[0], base_outs[1]  # each [1, T, H]

        # 5. Mask-mix: FSQ on audio positions, raw on text positions
        enc_out = fsq_out * am + raw_out * tm
        lm_hidden = enc_out[:, -1:, :]  # [1, 1, H]

        # 6. Residual LM prefill (includes fusion_proj — takes [1, T, 2*H])
        fusion_in = np.concatenate([enc_out, am * feat_embed], axis=-1)

        self._residual_lm.reset_state()
        res_outs = self._infer_stateful(
            self._residual_lm,
            fusion_in,
            attn_mask,
            pos_ids,
            n_out=self._residual_lm_n_out,
        )
        res_hidden = res_outs[0][:, -1:, :]  # [1, 1, H]

        # 7. Determine continuation context
        has_cont = audio_mask[0, -1] == 1
        context_len = 0
        prefix_feat_cond = audio_feat[:, -1, :, :]  # [1, P, D]
        pred_feat_seq = []

        if has_cont:
            a_indices = np.where(audio_mask[0] == 1)[0]
            context_len = min(streaming_prefix_len - 1, len(a_indices))
            last_idx = a_indices[-context_len:]
            for idx in last_idx:
                pred_feat_seq.append(audio_feat[:, idx : idx + 1, :, :])

        position = T  # current position counter

        # 8. Autoregressive loop
        for i in range(max_len):
            # Decode heads: dit_projections + stop_predictor in one call
            lm_h = lm_hidden.reshape(1, self.hidden_size)
            res_h = res_hidden.reshape(1, self.hidden_size)
            heads_out = self._decode_heads({"lm_hidden": lm_h, "res_hidden": res_h})
            dit_hidden = heads_out[0]  # [1, 2*H_dit]
            stop_logits = heads_out[1]  # [1, 2]

            # CFM Euler solve
            cond = prefix_feat_cond.transpose(0, 2, 1).copy()  # [1, D, P]
            pred_feat_raw = self._cfm_solve(dit_hidden, cond, inference_timesteps, cfg_value)
            pred_feat = pred_feat_raw.transpose(0, 2, 1)  # [1, P, D]

            # Encode predicted feature
            feat_for_enc = pred_feat.reshape(1, 1, P, D)
            curr_embed = self._feat_encoder({"audio_features": feat_for_enc})[0]  # [1, 1, H]

            pred_feat_seq.append(feat_for_enc)
            prefix_feat_cond = pred_feat  # [1, P, D]

            # Streaming: yield sliding window of last prefix_len patches
            if streaming:
                chunk_feats = pred_feat_seq[-streaming_prefix_len:]
                chunk = np.concatenate(chunk_feats, axis=1)  # [1, N, P, D]
                latent_chunk = chunk.transpose(0, 3, 1, 2).reshape(1, D, -1)
                yield latent_chunk, pred_feat_seq, context_len
                # Trim history to keep sliding window
                if len(pred_feat_seq) > streaming_prefix_len:
                    pred_feat_seq = pred_feat_seq[-streaming_prefix_len:]

            # Stop check
            if i > min_len and np.argmax(stop_logits[0]) == 1:
                break

            # Base LM decode step (returns [raw, fsq])
            attn_decode = np.ones([1, position + 1], dtype=np.int64)
            pos_decode = np.array([[position]], dtype=np.int64)
            base_dec = self._infer_stateful(
                self._base_lm,
                curr_embed[:, 0:1, :],
                attn_decode,
                pos_decode,
                n_out=self._base_lm_n_out,
            )
            lm_hidden = base_dec[1]  # use FSQ'd output

            # Residual LM decode step (takes [1, 1, 2*H] concatenated)
            res_input = np.concatenate([lm_hidden, curr_embed[:, 0:1, :]], axis=-1)
            res_dec = self._infer_stateful(
                self._residual_lm,
                res_input,
                attn_decode,
                pos_decode,
                n_out=self._residual_lm_n_out,
            )
            res_hidden = res_dec[0]  # [1, 1, H]

            position += 1

        # 9. Assemble latent for VAE decode (non-streaming only)
        if not streaming:
            all_feats = np.concatenate(pred_feat_seq, axis=1)  # [1, num_steps, P, D]
            latent = all_feats.transpose(0, 3, 1, 2).reshape(1, D, -1)
            generated = all_feats[:, context_len:, :, :]
            yield latent, generated, context_len

    # ── Sequence building ────────────────────────────────────────────

    def _build_sequence(
        self,
        text,
        reference_wav_path=None,
        prompt_wav_path=None,
        prompt_text=None,
    ):
        """Build text_token, text_mask, audio_feat, audio_mask arrays (with batch dim)."""
        P, D = self.patch_size, self.feat_dim

        if reference_wav_path and prompt_wav_path and prompt_text:
            full_text = prompt_text + text
            text_ids = self._tokenize(full_text)
            text_ids = np.append(text_ids, self.audio_start_token)
            text_length = len(text_ids)

            ref_feat = self.encode_wav(reference_wav_path, "right")
            prompt_feat = self.encode_wav(prompt_wav_path, "left")
            prompt_len = prompt_feat.shape[0]

            ref_tok, ref_f, ref_tm, ref_am = self._make_ref_prefix(ref_feat)

            pad_tok = np.zeros(prompt_len, dtype=np.int64)
            text_pad_f = np.zeros((text_length, P, D), dtype=np.float32)

            text_token = np.concatenate([ref_tok, text_ids, pad_tok])
            audio_feat = np.concatenate([ref_f, text_pad_f, prompt_feat], axis=0)
            text_mask = np.concatenate([ref_tm, np.ones(text_length, dtype=np.int32), np.zeros(prompt_len, dtype=np.int32)])
            audio_mask = np.concatenate([ref_am, np.zeros(text_length, dtype=np.int32), np.ones(prompt_len, dtype=np.int32)])

        elif reference_wav_path:
            text_ids = self._tokenize(text)
            text_ids = np.append(text_ids, self.audio_start_token)
            text_length = len(text_ids)

            ref_feat = self.encode_wav(reference_wav_path, "right")
            ref_tok, ref_f, ref_tm, ref_am = self._make_ref_prefix(ref_feat)

            text_pad_f = np.zeros((text_length, P, D), dtype=np.float32)
            text_token = np.concatenate([ref_tok, text_ids])
            audio_feat = np.concatenate([ref_f, text_pad_f], axis=0)
            text_mask = np.concatenate([ref_tm, np.ones(text_length, dtype=np.int32)])
            audio_mask = np.concatenate([ref_am, np.zeros(text_length, dtype=np.int32)])

        elif prompt_wav_path and prompt_text:
            full_text = prompt_text + text
            text_ids = self._tokenize(full_text)
            text_ids = np.append(text_ids, self.audio_start_token)
            text_length = len(text_ids)

            prompt_feat = self.encode_wav(prompt_wav_path, "left")
            prompt_len = prompt_feat.shape[0]

            pad_tok = np.zeros(prompt_len, dtype=np.int64)
            text_pad_f = np.zeros((text_length, P, D), dtype=np.float32)

            text_token = np.concatenate([text_ids, pad_tok])
            audio_feat = np.concatenate([text_pad_f, prompt_feat], axis=0)
            text_mask = np.concatenate([np.ones(text_length, dtype=np.int32), np.zeros(prompt_len, dtype=np.int32)])
            audio_mask = np.concatenate([np.zeros(text_length, dtype=np.int32), np.ones(prompt_len, dtype=np.int32)])

        else:
            text_ids = self._tokenize(text)
            text_ids = np.append(text_ids, self.audio_start_token)
            text_length = len(text_ids)

            text_token = text_ids
            audio_feat = np.zeros((text_length, P, D), dtype=np.float32)
            text_mask = np.ones(text_length, dtype=np.int32)
            audio_mask = np.zeros(text_length, dtype=np.int32)

        text_token = text_token.reshape(1, -1)
        text_mask = text_mask.reshape(1, -1)
        audio_feat = audio_feat[np.newaxis, ...]
        audio_mask = audio_mask.reshape(1, -1)
        return text_token, text_mask, audio_feat, audio_mask

    # ── Public generate method ────────────────────────────────────────

    def generate(
        self,
        text: str,
        reference_wav_path: str = None,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 2000,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech for the given text.

        Returns:
            (waveform, sample_rate) — waveform is 1-D float32 numpy array.
        """
        text_token, text_mask, audio_feat, audio_mask = self._build_sequence(
            text,
            reference_wav_path,
            prompt_wav_path,
            prompt_text,
        )

        latent, _, context_len = next(
            self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=max_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            )
        )

        audio = self.decode_latent(latent)
        decode_patch_len = self.patch_size * self.decode_chunk_size
        if context_len > 0:
            audio = audio[..., decode_patch_len * context_len :]

        wav = audio.squeeze()
        return wav, self.sample_rate

    # ── Public streaming generate ─────────────────────────────────────

    def generate_streaming(
        self,
        text: str,
        reference_wav_path: str = None,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 2000,
        streaming_prefix_len: int = 4,
    ) -> Generator[np.ndarray, None, None]:
        """Generate speech in streaming mode, yielding audio chunks.

        Each chunk is a 1-D float32 numpy array containing ~``decode_patch_len``
        samples (= ``patch_size * decode_chunk_size`` = 3840 samples at 48 kHz ≈ 80 ms).

        Yields:
            np.ndarray — 1-D float32 PCM waveform chunk.
        """
        text_token, text_mask, audio_feat, audio_mask = self._build_sequence(
            text,
            reference_wav_path,
            prompt_wav_path,
            prompt_text,
        )

        decode_patch_len = self.patch_size * self.decode_chunk_size

        for latent_chunk, _, _ctx in self._inference(
            text_token,
            text_mask,
            audio_feat,
            audio_mask,
            min_len=min_len,
            max_len=max_len,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            streaming=True,
            streaming_prefix_len=streaming_prefix_len,
        ):
            audio = self.decode_latent(latent_chunk)  # [1, 1, audio_len]
            audio = audio[..., -decode_patch_len:]  # last chunk only
            yield audio.squeeze()
