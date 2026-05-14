import sys

sys.path.append("CosyVoice")
sys.path.append("CosyVoice/third_party/Matcha-TTS")

# Disable torchao to avoid version conflict with torch 2.3.1
import os

os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"

import openvino as ov
import nncf  # type: ignore
from pathlib import Path
import torch
import torch.nn.functional as F
import random
import re
import time
import threading
import types
from typing import Callable, Generator
import openvino.opset13 as opset13
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
import numpy as np
import gc
from contextlib import nullcontext
import shutil
import json
from tqdm import tqdm
from transformers.cache_utils import DynamicCache
from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore

# OpenVINO core
core = ov.Core()


def patch_cos_sin_cached_fp32(model):
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
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    gather_dim: int,
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`list[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`list[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
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
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    key_value_output_names: list[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`list[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`list[str]`):
            list of names for key value input layers
        key_value_output_names (`list[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
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
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model, dim=1):
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


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    # Ethan
    # batch_size = lengths.size(0)
    # max_len = max_len if max_len > 0 else lengths.max().item()

    batch_size = lengths.shape[0]
    max_len = max_len if max_len > 0 else lengths.max()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


core = ov.Core()


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


TEXT_EMBEDDINGS_PATH = "openvino_text_embeddings_model.xml"
SPEECH_EMBEDDINGS_PATH = "openvino_speech_embeddings_model.xml"
LANGUAGE_PATH = "openvino_model.xml"
FLOW_EMBEDDINGS_PATH = "openvino_flow_embeddings_model.xml"
FLOW_ESTIMATOR_PATH = "openvino_flow_estimator_model.xml"
HIFT_PATH = "openvino_hift_model.xml"

# Dependency files to copy from original model directory
DEPENDENCY_FILES = [
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "spk2info.pt",
    "cosyvoice3.yaml",
]
# Dependency directories to copy
DEPENDENCY_DIRS = [
    "CosyVoice-BlankEN",  # Qwen tokenizer
]


def convert_cosyvoice(model_id, model_path=None, quantization_config=None):

    if model_path is None:
        model_path = Path(model_id.split("/")[-1])
    else:
        model_path = Path(model_path)

    if all(
        (model_path / model_name).exists()
        for model_name in [
            SPEECH_EMBEDDINGS_PATH,
            TEXT_EMBEDDINGS_PATH,
            SPEECH_EMBEDDINGS_PATH,
            LANGUAGE_PATH,
            FLOW_EMBEDDINGS_PATH,
            FLOW_ESTIMATOR_PATH,
            HIFT_PATH,
        ]
    ):
        print(f"✅ {model_id} model already converted. You can find results in {model_path}")
        return model_path
    print(f"⌛ {model_id} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")
    pt_model = AutoModel(model_dir=model_id)
    print("✅ Original model successfully loaded")

    if not (model_path / TEXT_EMBEDDINGS_PATH).exists():
        print("⌛ Convert TEXT_EMBEDDINGS model")

        ov_model = ov.convert_model(pt_model.model.llm.llm.model.model.embed_tokens, example_input=torch.ones([1, 43], dtype=torch.int32))
        ov.save_model(ov_model, model_path / TEXT_EMBEDDINGS_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ TEXT_EMBEDDINGS model successfully converted")

    if not (model_path / SPEECH_EMBEDDINGS_PATH).exists():
        print("⌛ Convert SPEECH_EMBEDDINGS model")

        ov_model = ov.convert_model(pt_model.model.llm.speech_embedding, example_input=torch.ones([1, 87], dtype=torch.int32))
        ov.save_model(ov_model, model_path / SPEECH_EMBEDDINGS_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ SPEECH_EMBEDDINGS model successfully converted")

    if not (model_path / FLOW_EMBEDDINGS_PATH).exists():
        print("⌛ Convert FLOW embeddings model")

        def forward_wrap_flow_emb(
            self,
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            #   prompt_feat,
            embedding,
        ):
            # Normalize and project speaker embedding
            embedding = F.normalize(embedding, dim=1)
            spks = self.spk_embed_affine_layer(embedding)  # (batch, 80)

            # concat text and prompt_text
            token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
            mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(spks)
            token = self.input_embedding(torch.clamp(token, min=0)) * mask

            h = self.pre_lookahead_layer(token)

            # Return both h and projected speaker embedding
            return h, spks

        pt_model.model.flow._orig_forward = pt_model.model.flow.forward
        pt_model.model.flow.forward = types.MethodType(forward_wrap_flow_emb, pt_model.model.flow)
        example_input = {
            "token": torch.ones([1, 94], dtype=torch.int32),
            "token_len": torch.tensor([94]).to(dtype=torch.int32),
            "prompt_token": torch.ones([1, 43], dtype=torch.int32),
            "prompt_token_len": torch.tensor([43]).to(dtype=torch.int32),
            # "prompt_feat": torch.ones([1, 87, 80], dtype=torch.float32),
            "embedding": torch.ones([1, 192], dtype=torch.float32),
        }

        ov_model = ov.convert_model(pt_model.model.flow, example_input=example_input)
        ov.save_model(ov_model, model_path / FLOW_EMBEDDINGS_PATH)
        del ov_model
        pt_model.model.flow.forward = pt_model.model.flow._orig_forward
        del pt_model.model.flow._orig_forward
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ FLOW embeddings model successfully converted")

    if not (model_path / FLOW_ESTIMATOR_PATH).exists():
        print("⌛ Convert FLOW estimator model")

        # Patch AttnProcessor to convert boolean mask to float mask for SDPA optimization
        # This enables OpenVINO GPU to use fused SDPA kernel instead of decomposed attention
        from cosyvoice.flow.DiT.modules import AttnProcessor, JointAttnProcessor

        _orig_attn_call = AttnProcessor.__call__

        def patched_attn_call(self, attn, x, mask=None, rope=None):
            batch_size = x.shape[0]
            query = attn.to_q(x)
            key = attn.to_k(x)
            value = attn.to_v(x)

            if rope is not None:
                from x_transformers.x_transformers import apply_rotary_pos_emb

                freqs, xpos_scale = rope
                q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if mask is not None:
                attn_mask = mask
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
                    attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
                # Convert boolean mask to float mask for SDPA optimization
                if attn_mask.dtype == torch.bool:
                    attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill(~attn_mask, float("-inf"))
            else:
                attn_mask = None

            x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            x = x.to(query.dtype)
            x = attn.to_out[0](x)
            x = attn.to_out[1](x)

            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)
                else:
                    mask = mask[:, 0, -1].unsqueeze(-1)
                x = x.masked_fill(~mask, 0.0)
            return x

        _orig_joint_call = JointAttnProcessor.__call__

        def patched_joint_call(self, attn, x, c=None, mask=None, rope=None, c_rope=None):
            from x_transformers.x_transformers import apply_rotary_pos_emb

            residual = x
            batch_size = c.shape[0]

            query = attn.to_q(x)
            key = attn.to_k(x)
            value = attn.to_v(x)
            c_query = attn.to_q_c(c)
            c_key = attn.to_k_c(c)
            c_value = attn.to_v_c(c)

            if rope is not None:
                freqs, xpos_scale = rope
                q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
            if c_rope is not None:
                freqs, xpos_scale = c_rope
                q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
                c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
                c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

            query = torch.cat([query, c_query], dim=1)
            key = torch.cat([key, c_key], dim=1)
            value = torch.cat([value, c_value], dim=1)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if mask is not None:
                attn_mask = F.pad(mask, (0, c.shape[1]), value=True)
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
                # Convert boolean mask to float mask for SDPA optimization
                if attn_mask.dtype == torch.bool:
                    attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill(~attn_mask, float("-inf"))
            else:
                attn_mask = None

            x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            x = x.to(query.dtype)

            x, c = x[:, : residual.shape[1]], x[:, residual.shape[1] :]
            x = attn.to_out[0](x)
            x = attn.to_out[1](x)
            if not attn.context_pre_only:
                c = attn.to_out_c(c)

            if mask is not None:
                mask = mask.unsqueeze(-1)
                x = x.masked_fill(~mask, 0.0)
            return x, c

        # Apply patches
        AttnProcessor.__call__ = patched_attn_call
        JointAttnProcessor.__call__ = patched_joint_call
        print("  Applied SDPA optimization patch (bool mask -> float mask)")

        example_input = {
            "x": torch.ones([2, 80, 634], dtype=torch.float32),
            "mask": torch.ones([2, 1, 634], dtype=torch.float32),
            "mu": torch.ones([2, 80, 634], dtype=torch.float32),
            "t": torch.ones([2], dtype=torch.float32),
            "spks": torch.ones([2, 80], dtype=torch.float32),
            "cond": torch.ones([2, 80, 634], dtype=torch.float32),
        }
        ov_model = ov.convert_model(pt_model.model.flow.decoder.estimator, example_input=example_input)

        # Restore original methods
        AttnProcessor.__call__ = _orig_attn_call
        JointAttnProcessor.__call__ = _orig_joint_call

        ov.save_model(ov_model, model_path / FLOW_ESTIMATOR_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ FLOW_ESTIMATOR model successfully converted")

    if not (model_path / HIFT_PATH).exists():
        print("⌛ Convert HIFT model (without istft post-processing)")

        # Export only the neural network part, stop before istft
        # This removes: exp, sin, _istft, clamp operations
        # Output will be raw conv_post output of shape (batch, n_fft+2, time)
        def forward_wrap_hift_no_istft(self, x: torch.Tensor) -> torch.Tensor:
            """
            HiFT forward that includes f0_predictor and m_source, but outputs before istft.
            Input: mel spectrogram (batch, 80, mel_len)
            Output: raw spectral features (batch, n_fft+2, time) before istft
            """
            # mel -> f0
            f0 = self.f0_predictor(x)
            # f0 -> source
            s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
            s, _, _ = self.m_source(s)
            s = s.transpose(1, 2)

            # Run decode up to conv_post only
            s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
            s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

            x = self.conv_pre(x)
            for i in range(self.num_upsamples):
                x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
                x = self.ups[i](x)

                if i == self.num_upsamples - 1:
                    x = self.reflection_pad(x)

                # fusion
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

            x = torch.nn.functional.leaky_relu(x)
            x = self.conv_post(x)
            # Return here without exp, sin, istft, clamp
            return x

        pt_model.model.hift._orig_forward = pt_model.model.hift.forward
        pt_model.model.hift.forward = types.MethodType(forward_wrap_hift_no_istft, pt_model.model.hift)
        ov_model = ov.convert_model(pt_model.model.hift, example_input=torch.ones([1, 80, 488], dtype=torch.float32))

        ov.save_model(ov_model, model_path / HIFT_PATH)
        del ov_model
        pt_model.model.hift.forward = pt_model.model.hift._orig_forward
        del pt_model.model.hift._orig_forward
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ HIFT model successfully converted")
    if not (model_path / LANGUAGE_PATH).exists():

        print("⌛ Convert LANGUAGE_MODEL model")
        patch_cos_sin_cached_fp32(pt_model.model.llm.llm)
        if hasattr(pt_model.model.llm.llm, "model"):
            patch_cos_sin_cached_fp32(pt_model.model.llm.llm.model)

        def forward_wrap(
            self,
            attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
        ):
            if past_key_values is not None:
                pkv = DynamicCache.from_legacy_cache(past_key_values)
            outs = self.llm.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
                past_key_values=pkv,
            )
            xs = outs.hidden_states[-1]
            new_cache = outs.past_key_values
            logp = self.llm_decoder(xs[:, -1])
            return (logp, new_cache.to_legacy_cache())

        pt_model.model.llm._orig_forward = pt_model.model.llm.forward
        pt_model.model.llm.forward = types.MethodType(forward_wrap, pt_model.model.llm)

        num_pkv = pt_model.model.llm.llm.model.config.num_hidden_layers
        hidden_size = pt_model.model.llm.llm.model.config.hidden_size
        head_dim = (
            pt_model.model.llm.llm.model.config.head_dim
            if hasattr(pt_model.model.llm.llm.model.config, "head_dim")
            else (hidden_size // pt_model.model.llm.llm.model.config.num_attention_heads)
        )
        pkv_shape = (
            2,
            pt_model.model.llm.llm.model.config.num_key_value_heads,
            2,
            head_dim,
        )

        inputs_embeds = torch.randn((2, 2, hidden_size))
        attention_mask = torch.ones([2, 4], dtype=torch.int64)
        position_ids = torch.arange(2).unsqueeze(0).expand(2, -1)

        input_names = ["attention_mask", "position_ids"]
        output_names = ["logits"]
        past_key_values = []
        for i in range(num_pkv):
            kv = [torch.randn(pkv_shape) for _ in range(2)]
            past_key_values.append(kv)
            input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        input_names.extend(["inputs_embeds"])
        example_input = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
        }

        input_shapes = [
            ov.PartialShape([-1, -1]),  # attention_mask
            ov.PartialShape([-1, -1]),  # position_ids (2D for code predictor)
        ]
        input_shapes += (
            [
                ov.PartialShape(
                    [
                        -1,
                        pt_model.model.llm.llm.model.config.num_key_value_heads,
                        -1,
                        head_dim,
                    ]
                )
            ]
            * 2
            * num_pkv
        )
        input_shapes += [ov.PartialShape([-1, -1, hidden_size])]  # inputs_embeds
        __make_16bit_traceable(pt_model.model.llm)

        ov_model = ov.convert_model(pt_model.model.llm, example_input=example_input, input=input_shapes)
        for input, input_name in zip(ov_model.inputs, input_names):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})
        patch_stateful(ov_model)
        print("✅ Decoder model successfully converted")
        if quantization_config is not None and "llm" in quantization_config:
            print(f"⌛ Weights compression with {quantization_config['llm']['mode']} mode started")
            ov_model = nncf.compress_weights(ov_model, **quantization_config["llm"])
            print("✅ Weights compression finished")
        else:
            ov_model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
        ov.save_model(ov_model, model_path / LANGUAGE_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()

    # Copy dependency files from original model directory
    model_id_path = Path(model_id)
    print("⌛ Copying dependency files...")

    # Copy individual files
    for dep_file in DEPENDENCY_FILES:
        src_path = model_id_path / dep_file
        dst_path = model_path / dep_file
        if src_path.exists() and not dst_path.exists():
            print(f"  Copying {dep_file}...")
            shutil.copy2(src_path, dst_path)

    # Copy directories (exclude .safetensors files)
    def ignore_safetensors(dir, files):
        return [f for f in files if f.endswith(".safetensors")]

    for dep_dir in DEPENDENCY_DIRS:
        src_dir = model_id_path / dep_dir
        dst_dir = model_path / dep_dir
        if src_dir.exists() and not dst_dir.exists():
            print(f"  Copying directory {dep_dir} (excluding .safetensors files)...")
            shutil.copytree(src_dir, dst_dir, ignore=ignore_safetensors)

    print("✅ Dependency files copied")

    del pt_model
    gc.collect()
    print(f"✅ {model_id} model conversion finished. You can find results in {model_path}")
    return model_path


class OVCosyVoice3LM:
    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        speech_token_size: int = 6561,
        llm_input_size: int = 896,
        npu_ov_config: dict = None,
    ):
        """
        Initialize OVCosyVoice3LM with OpenVINO models.

        Args:
            model_path: Path to the directory containing converted OpenVINO models
            device: OpenVINO device (CPU, GPU, etc.)
            speech_token_size: Size of speech token vocabulary
            llm_input_size: Input size of the LLM
        """
        self.model_path = Path(model_path)
        self.ov_device = device
        self.speech_token_size = speech_token_size
        self.llm_input_size = llm_input_size

        # Token IDs (same as CosyVoice3LM)
        self.sos = speech_token_size + 0
        self.eos_token = speech_token_size + 1
        self.task_id = speech_token_size + 2
        self.fill_token = speech_token_size + 3

        # Load OpenVINO models
        print("⌛ Loading OpenVINO models...")

        # Text embeddings model
        self.text_embeddings = core.compile_model(self.model_path / TEXT_EMBEDDINGS_PATH, device if device != "NPU" else "GPU")
        print(f"✅ Text embeddings model loaded")

        # Speech embeddings model
        self.speech_embeddings = core.compile_model(self.model_path / SPEECH_EMBEDDINGS_PATH, device if device != "NPU" else "GPU")
        print(f"✅ Speech embeddings model loaded")

        # LLM model (stateful)
        self.llm = core.compile_model(self.model_path / LANGUAGE_PATH, device, npu_ov_config if device == "NPU" else {})
        self.llm_request = self.llm.create_infer_request()
        print(f"✅ LLM model loaded")

        # Stop token IDs for generation (same as CosyVoice3LM)
        self.stop_token_ids = [speech_token_size + i for i in range(200)]

    def embed_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Embed text tokens using OpenVINO model."""
        result = self.text_embeddings(text_tokens.numpy() if isinstance(text_tokens, torch.Tensor) else text_tokens)
        return torch.from_numpy(result[0])

    def embed_speech(self, speech_tokens: torch.Tensor) -> torch.Tensor:
        """Embed speech tokens using OpenVINO model."""
        result = self.speech_embeddings(speech_tokens.numpy() if isinstance(speech_tokens, torch.Tensor) else speech_tokens)
        return torch.from_numpy(result[0])

    def get_sos_emb(self) -> torch.Tensor:
        """Get SOS embedding."""
        sos_token = torch.tensor([[self.sos]], dtype=torch.int32)
        return self.embed_speech(sos_token)

    def get_task_id_emb(self) -> torch.Tensor:
        """Get task ID embedding."""
        task_id_token = torch.tensor([[self.task_id]], dtype=torch.int32)
        return self.embed_speech(task_id_token)

    def reset_state(self):
        """Reset LLM state for new generation."""
        self.llm_request.reset_state()

    def forward_one_step(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run one step of LLM inference.

        Args:
            inputs_embeds: Input embeddings (batch, seq_len, hidden_size)
            attention_mask: Attention mask (batch, total_seq) - 2D mask for NPU
            position_ids: Position IDs (batch, seq_len)

        Returns:
            logits: Output logits for next token prediction
        """
        inputs = {
            "inputs_embeds": inputs_embeds.numpy() if isinstance(inputs_embeds, torch.Tensor) else inputs_embeds,
            "attention_mask": attention_mask.numpy() if isinstance(attention_mask, torch.Tensor) else attention_mask,
            "position_ids": position_ids.numpy() if isinstance(position_ids, torch.Tensor) else position_ids,
        }

        # Add beam_idx for stateful model
        batch_size = inputs_embeds.shape[0]
        if "beam_idx" in [inp.get_any_name() for inp in self.llm.inputs]:
            inputs["beam_idx"] = np.arange(batch_size, dtype=np.int32)

        self.llm_request.infer(inputs)
        logits = torch.from_numpy(self.llm_request.get_tensor("logits").data.copy())
        return logits

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        decoded_tokens: list,
        sampling: int,
        ignore_eos: bool = True,
    ):
        """Sample token IDs from logits (same as original CosyVoice3LM)."""
        while True:
            top_ids = weighted_scores.multinomial(sampling, replacement=True)
            top_id = top_ids[random.randint(0, sampling - 1)]  # nosec B311 - ML sampling index, not security
            if (not ignore_eos) or (ignore_eos and top_id not in self.stop_token_ids):
                break
        return top_id.item()

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = "",
    ):
        """
        Inference method matching CosyVoice3LM interface.

        Args:
            text: Input text tokens (batch, seq_len)
            text_len: Length of text tokens
            prompt_text: Prompt text tokens
            prompt_text_len: Length of prompt text tokens
            prompt_speech_token: Prompt speech tokens
            prompt_speech_token_len: Length of prompt speech tokens
            embedding: Speaker embedding (not used in OpenVINO version)
            sampling: Sampling parameter for token selection
            max_token_text_ratio: Maximum ratio of generated tokens to text tokens
            min_token_text_ratio: Minimum ratio of generated tokens to text tokens
            uuid: Unique identifier for this generation

        Yields:
            Generated speech tokens one by one
        """
        # Concatenate prompt_text and text
        text = torch.concat([prompt_text, text], dim=1)
        text_len = text_len + prompt_text_len

        # Embed text tokens using OpenVINO model
        text_emb = self.embed_text(text.to(torch.int32))

        # Get special token embeddings
        sos_emb = self.get_sos_emb()
        task_id_emb = self.get_task_id_emb()

        # Embed prompt speech tokens if provided
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.embed_speech(prompt_speech_token.to(torch.int32))
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text_emb.dtype)

        # Concatenate all inputs: [sos, text, task_id, prompt_speech]
        lm_input = torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)

        # Calculate min/max generation length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)
        # Step by step decode
        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
            yield token

    @torch.inference_mode()
    def inference_wrapper(self, lm_input: torch.Tensor, sampling: int, min_len: int, max_len: int, uuid: str = ""):
        """
        Wrapper for autoregressive generation using OpenVINO.

        Args:
            lm_input: Initial input embeddings (batch, seq_len, hidden_size)
            sampling: Sampling parameter for token selection
            min_len: Minimum number of tokens to generate
            max_len: Maximum number of tokens to generate
            uuid: Unique identifier for this generation

        Yields:
            Generated speech tokens one by one
        """
        # Reset state for new generation
        self.reset_state()

        out_tokens = []
        seq_len = lm_input.shape[1]
        current_pos = 0  # Track current position for position_ids

        for i in range(max_len):
            # Create attention mask and position_ids for current step
            # For NPU stateful model:
            # - attention_mask: 2D mask (batch, total_seq)
            # - position_ids: 2D (batch, current_seq)
            current_seq = lm_input.shape[1]
            if i == 0:
                # First step: full sequence
                total_seq = current_seq
                attention_mask = torch.ones((1, total_seq), dtype=torch.int64)
                position_ids = torch.arange(current_seq, dtype=torch.int64).unsqueeze(0)
                current_pos = current_seq
            else:
                # Subsequent steps: single new token
                total_seq = seq_len + len(out_tokens)
                attention_mask = torch.ones((1, total_seq), dtype=torch.int64)
                position_ids = torch.tensor([[current_pos]], dtype=torch.int64)
                current_pos += 1

            # Run one step of LLM
            logits = self.forward_one_step(lm_input, attention_mask, position_ids)

            # Get log probabilities and sample
            # Check for NaN/Inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN or Inf in logits at step {i}")
                break

            logp = logits.log_softmax(dim=-1)

            top_ids = self.sampling_ids(logp.squeeze(dim=0).exp(), out_tokens, sampling, ignore_eos=True if i < min_len else False)

            # Check for stop tokens
            if top_ids in self.stop_token_ids:
                break

            # Yield token in stream mode
            yield top_ids
            out_tokens.append(top_ids)

            # Prepare next input: embed the generated token
            next_token = torch.tensor([[top_ids]], dtype=torch.int32)
            lm_input = self.embed_speech(next_token)


class OVCosyVoiceFrontEnd:
    """
    OpenVINO-based CosyVoice FrontEnd for text and speech preprocessing.

    Uses OpenVINO for ONNX model inference (campplus, speech_tokenizer)
    instead of ONNX Runtime.
    """

    def __init__(
        self,
        get_tokenizer: Callable,
        feat_extractor: Callable,
        campplus_model: str,
        speech_tokenizer_model: str,
        spk2info: str = "",
        allowed_special: str = "all",
        device: str = "CPU",
    ):
        """
        Initialize OVCosyVoiceFrontEnd.

        Args:
            get_tokenizer: Function to get the text tokenizer
            feat_extractor: Feature extractor for speech
            campplus_model: Path to campplus ONNX model
            speech_tokenizer_model: Path to speech tokenizer ONNX model
            spk2info: Path to speaker info file
            allowed_special: Allowed special tokens
            device: OpenVINO device (CPU, GPU, etc.)
        """
        import whisper
        import torchaudio.compliance.kaldi as kaldi
        from cosyvoice.utils.file_utils import logging, load_wav

        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ov_device = device

        # Store utilities for later use
        self.whisper = whisper
        self.kaldi = kaldi
        self.load_wav = load_wav
        self.logging = logging

        # Load campplus model with OpenVINO (directly from ONNX)
        print(f"⌛ Loading OpenVINO campplus model from {campplus_model}...")
        self.campplus_model = core.compile_model(campplus_model, device)
        self.campplus_input_name = self.campplus_model.inputs[0].get_any_name()
        print(f"✅ Campplus model loaded")

        # Load speech tokenizer model with OpenVINO (directly from ONNX)
        print(f"⌛ Loading OpenVINO speech tokenizer model from {speech_tokenizer_model}...")
        self.speech_tokenizer_model = core.compile_model(speech_tokenizer_model, device)
        # Get input names for speech tokenizer
        self.speech_tokenizer_input_names = [inp.get_any_name() for inp in self.speech_tokenizer_model.inputs]
        print(f"✅ Speech tokenizer model loaded")

        # Load speaker info
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.torch_device)
        else:
            self.spk2info = {}

        self.allowed_special = allowed_special

        # Text normalization setup
        try:
            import ttsfrd

            self.use_ttsfrd = True
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize("{}/../../pretrained_models/CosyVoice-ttsfrd/resource".format(ROOT_DIR)) is True, "failed to initialize ttsfrd resource"
            self.frd.set_lang_type("pinyinvg")
        except ImportError:
            print("failed to import ttsfrd, use wetext instead")
            from wetext import Normalizer as ZhNormalizer
            from wetext import Normalizer as EnNormalizer
            import inflect

            self.use_ttsfrd = False
            self.zh_tn_model = ZhNormalizer(remove_erhua=False)
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()

    def _extract_text_token(self, text):
        """Extract text tokens using the tokenizer."""
        if isinstance(text, Generator):
            self.logging.info("get tts_text generator, will return _extract_text_token_generator!")
            return self._extract_text_token_generator(text), torch.tensor([0], dtype=torch.int32).to(self.torch_device)
        else:
            text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
            text_token = torch.tensor([text_token], dtype=torch.int32).to(self.torch_device)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.torch_device)
            return text_token, text_token_len

    def _extract_text_token_generator(self, text_generator):
        """Generator for extracting text tokens."""
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    def _extract_speech_token(self, prompt_wav):
        """Extract speech tokens using OpenVINO speech tokenizer."""
        speech = self.load_wav(prompt_wav, 16000)
        assert speech.shape[1] / 16000 <= 30, "do not support extract speech token for audio longer than 30s"

        # Get mel spectrogram using whisper
        feat = self.whisper.log_mel_spectrogram(speech, n_mels=128)

        # Prepare inputs for OpenVINO
        feat_np = feat.detach().cpu().numpy()
        feat_len_np = np.array([feat.shape[2]], dtype=np.int32)

        # Run OpenVINO inference
        inputs = {self.speech_tokenizer_input_names[0]: feat_np, self.speech_tokenizer_input_names[1]: feat_len_np}
        result = self.speech_tokenizer_model(inputs)
        speech_token = result[0].flatten().tolist()

        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.torch_device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.torch_device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, prompt_wav):
        """Extract speaker embedding using OpenVINO campplus model."""
        speech = self.load_wav(prompt_wav, 16000)

        # Extract fbank features
        feat = self.kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)

        # Prepare input for OpenVINO
        feat_np = feat.unsqueeze(dim=0).cpu().numpy()

        # Run OpenVINO inference
        result = self.campplus_model({self.campplus_input_name: feat_np})
        embedding = result[0].flatten().tolist()

        embedding = torch.tensor([embedding]).to(self.torch_device)
        return embedding

    def _extract_speech_feat(self, prompt_wav):
        """Extract speech features (mel spectrogram)."""
        speech = self.load_wav(prompt_wav, 24000)
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.torch_device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.torch_device)
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True, text_frontend=True):
        """Normalize text for TTS."""
        from functools import partial
        from cosyvoice.utils.frontend_utils import (
            contains_chinese,
            replace_blank,
            replace_corner_mark,
            remove_bracket,
            spell_out_number,
            split_paragraph,
            is_only_punctuation,
        )

        if isinstance(text, Generator):
            self.logging.info("get tts_text generator, will skip text_normalize!")
            return [text]

        # Skip text_frontend when ssml symbol in text
        if "<|" in text and "|>" in text:
            text_frontend = False
        if text_frontend is False or text == "":
            return [text] if split is True else text

        text = text.strip()
        if self.use_ttsfrd:
            texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
            text = "".join(texts)
        else:
            if contains_chinese(text):
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r"[，,、]+$", "。", text)
                texts = list(
                    split_paragraph(
                        text,
                        partial(self.tokenizer.encode, allowed_special=self.allowed_special),
                        "zh",
                        token_max_n=80,
                        token_min_n=60,
                        merge_len=20,
                        comma_split=False,
                    )
                )
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(
                    split_paragraph(
                        text,
                        partial(self.tokenizer.encode, allowed_special=self.allowed_special),
                        "en",
                        token_max_n=80,
                        token_min_n=60,
                        merge_len=20,
                        comma_split=False,
                    )
                )

        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def frontend_sft(self, tts_text, spk_id):
        """Prepare model input for SFT inference."""
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]["embedding"]
        model_input = {"text": tts_text_token, "text_len": tts_text_token_len, "llm_embedding": embedding, "flow_embedding": embedding}
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id):
        """Prepare model input for zero-shot inference."""
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        if zero_shot_spk_id == "":
            prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_wav)
            speech_token, speech_token_len = self._extract_speech_token(prompt_wav)
            if resample_rate == 24000:
                # cosyvoice2/3, force speech_feat % speech_token = 2
                token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
                speech_feat, speech_feat_len[:] = speech_feat[:, : 2 * token_len], 2 * token_len
                speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
            embedding = self._extract_spk_embedding(prompt_wav)
            model_input = {
                "prompt_text": prompt_text_token,
                "prompt_text_len": prompt_text_token_len,
                "llm_prompt_speech_token": speech_token,
                "llm_prompt_speech_token_len": speech_token_len,
                "flow_prompt_speech_token": speech_token,
                "flow_prompt_speech_token_len": speech_token_len,
                "prompt_speech_feat": speech_feat,
                "prompt_speech_feat_len": speech_feat_len,
                "llm_embedding": embedding,
                "flow_embedding": embedding,
            }
        else:
            model_input = self.spk2info[zero_shot_spk_id]
        model_input["text"] = tts_text_token
        model_input["text_len"] = tts_text_token_len
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_wav, resample_rate, zero_shot_spk_id):
        """Prepare model input for cross-lingual inference."""
        model_input = self.frontend_zero_shot(tts_text, "", prompt_wav, resample_rate, zero_shot_spk_id)
        # in cross lingual mode, we remove prompt in llm
        del model_input["prompt_text"]
        del model_input["prompt_text_len"]
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        """Prepare model input for instruct inference."""
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input["llm_embedding"]
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text)
        model_input["prompt_text"] = instruct_text_token
        model_input["prompt_text_len"] = instruct_text_token_len
        return model_input

    def frontend_instruct2(self, tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id):
        """Prepare model input for instruct2 inference."""
        model_input = self.frontend_zero_shot(tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id)
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_wav, resample_rate):
        """Prepare model input for voice conversion."""
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(prompt_wav)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_wav)
        embedding = self._extract_spk_embedding(prompt_wav)
        source_speech_token, source_speech_token_len = self._extract_speech_token(source_speech_16k)
        model_input = {
            "source_speech_token": source_speech_token,
            "source_speech_token_len": source_speech_token_len,
            "flow_prompt_speech_token": prompt_speech_token,
            "flow_prompt_speech_token_len": prompt_speech_token_len,
            "prompt_speech_feat": prompt_speech_feat,
            "prompt_speech_feat_len": prompt_speech_feat_len,
            "flow_embedding": embedding,
        }
        return model_input


class OVFlow:
    """
    OpenVINO-based Flow model for mel spectrogram generation.

    Uses two OpenVINO models:
    - flow_embeddings: Processes token embedding and pre_lookahead_layer
    - flow_estimator: DiT model for flow matching denoising
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "CPU",
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        output_size: int = 80,
        n_timesteps: int = 10,
        sigma_min: float = 1e-6,
        inference_cfg_rate: float = 0.7,
    ):
        """
        Initialize OVFlow with OpenVINO models.

        Args:
            model_dir: Path to the directory containing OpenVINO flow models
            device: OpenVINO device (CPU, GPU, etc.)
            token_mel_ratio: Ratio of mel frames per token
            pre_lookahead_len: Lookahead length for streaming
            output_size: Output mel dimension (default 80)
            n_timesteps: Number of diffusion steps (default 10)
            sigma_min: Minimum sigma for flow matching
            inference_cfg_rate: Classifier-free guidance rate
        """
        self.model_dir = Path(model_dir)
        self.ov_device = device

        # Flow parameters
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
        self.output_size = output_size
        self.n_timesteps = n_timesteps
        self.sigma_min = sigma_min
        self.inference_cfg_rate = inference_cfg_rate

        # Load OpenVINO flow embeddings model
        flow_emb_path = self.model_dir / FLOW_EMBEDDINGS_PATH
        print(f"⌛ Loading OpenVINO Flow embeddings model from {flow_emb_path}...")
        self.flow_embeddings = core.compile_model(str(flow_emb_path), "CPU")
        print(f"✅ Flow embeddings model loaded")

        # Load OpenVINO flow estimator model (DiT)
        flow_est_path = self.model_dir / FLOW_ESTIMATOR_PATH
        print(f"⌛ Loading OpenVINO Flow estimator model from {flow_est_path}...")
        self.flow_estimator = core.compile_model(str(flow_est_path), device)
        print(f"✅ Flow estimator model loaded")

        # Pre-generate random noise for deterministic inference
        self._init_rand_noise()

    def _init_rand_noise(self, max_len: int = 50 * 300):
        """Initialize random noise buffer for deterministic inference."""
        torch.manual_seed(0)
        self.rand_noise = torch.randn([1, self.output_size, max_len])

    def _run_flow_embeddings(self, token, token_len, prompt_token, prompt_token_len, embedding):
        """
        Run flow embeddings model to get hidden states.

        Args:
            token: Speech tokens (batch, seq_len)
            token_len: Token lengths
            prompt_token: Prompt speech tokens
            prompt_token_len: Prompt token lengths
            embedding: Speaker embedding (batch, 192)

        Returns:
            h: Hidden states from pre_lookahead_layer (batch, seq_len, hidden_size)
            spks: Projected speaker embedding (batch, 80)
        """
        inputs = {
            "token": token,
            "token_len": token_len,
            "prompt_token": prompt_token,
            "prompt_token_len": prompt_token_len,
            "embedding": embedding,
        }
        result = self.flow_embeddings(inputs)
        h = torch.from_numpy(result[0].copy())
        spks = torch.from_numpy(result[1].copy())
        return h, spks

    def _run_flow_estimator(self, x, mask, mu, t, spks, cond):
        """
        Run flow estimator (DiT) model for one denoising step.

        Args:
            x: Noised input (batch, 80, mel_len)
            mask: Output mask (batch, 1, mel_len)
            mu: Encoder output / condition (batch, 80, mel_len)
            t: Timestep (batch,)
            spks: Speaker embedding (batch, 80)
            cond: Conditioning (batch, 80, mel_len)

        Returns:
            Estimated velocity field (batch, 80, mel_len)
        """
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
        Euler ODE solver for flow matching.

        Args:
            z: Initial noise (batch, 80, mel_len)
            t_span: Time steps (n_timesteps + 1,)
            mu: Encoder output (batch, 80, mel_len)
            mask: Output mask (batch, 1, mel_len)
            spks: Speaker embedding (batch, 80)
            cond: Conditioning (batch, 80, mel_len)

        Returns:
            Final sample (batch, 80, mel_len)
        """
        x = z
        t, dt = t_span[0], t_span[1] - t_span[0]

        # Prepare batched inputs for CFG (classifier-free guidance)
        # Batch size 2: [with_condition, without_condition]
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

    def inference(self, token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, streaming=False, finalize=True):
        """
        Run flow inference using OpenVINO.

        Args:
            token: Speech tokens (batch, seq_len)
            token_len: Token lengths
            prompt_token: Prompt speech tokens
            prompt_token_len: Prompt token lengths
            prompt_feat: Prompt mel features (batch, seq_len, 80)
            prompt_feat_len: Prompt feature lengths
            embedding: Speaker embedding (batch, 192)
            streaming: Whether streaming mode (not used in OV version)
            finalize: Whether this is the final chunk (not used in OV version)

        Returns:
            mel: Generated mel spectrogram (batch, 80, mel_len)
            None: Placeholder for cache
        """
        # Step 1: Run flow embeddings to get hidden states and projected speaker embedding
        # This does: embedding normalization, token concat, input_embedding, pre_lookahead_layer
        h, spks = self._run_flow_embeddings(
            token.to(torch.int32), token_len.to(torch.int32), prompt_token.to(torch.int32), prompt_token_len.to(torch.int32), embedding.to(torch.float32)
        )

        # Step 2: Repeat interleave for token_mel_ratio (2x upsampling)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)

        # Calculate mel lengths
        mel_len1 = prompt_feat.shape[1]  # prompt mel length
        mel_len2 = h.shape[1] - mel_len1  # generated mel length
        total_mel_len = mel_len1 + mel_len2

        # Step 3: Prepare conditions
        # conds: (batch, mel_len, 80) -> (batch, 80, mel_len)
        conds = torch.zeros([1, total_mel_len, self.output_size], device=h.device, dtype=h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)  # (batch, 80, mel_len)

        # Step 4: Prepare mask
        mask = torch.ones([1, 1, total_mel_len], dtype=h.dtype)

        # Step 5: Prepare mu (encoder output)
        mu = h.transpose(1, 2).contiguous()  # (batch, 80, mel_len)

        # Step 6: Speaker embedding is already projected by flow_embeddings (batch, 80)
        spks = spks.to(h.dtype)

        # Step 7: Initialize noise
        z = self.rand_noise[:, :, :total_mel_len].to(h.device).to(h.dtype)

        # Step 8: Create time span with cosine schedule
        t_span = torch.linspace(0, 1, self.n_timesteps + 1, dtype=h.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)  # Cosine schedule

        # Step 9: Solve ODE with Euler method
        feat = self._solve_euler(z, t_span, mu, mask, spks, conds)

        # Step 10: Return only the generated part (exclude prompt)
        feat = feat[:, :, mel_len1:]

        return feat, None


class OVHiFT:
    """OpenVINO-based HiFT vocoder for waveform generation."""

    def __init__(self, model_path: str, device: str = "CPU", hift_input_len: int = 0):
        """
        Initialize OVHiFT with OpenVINO model.

        Args:
            model_path: Path to the OpenVINO hift model (.xml)
            device: OpenVINO device (CPU, GPU, etc.)
            hift_input_len: Fixed input length for HiFT model. If > 0, model is reshaped to this length.
        """
        self.model_path = Path(model_path)
        self.ov_device = device
        self.hift_input_len = hift_input_len
        # Load OpenVINO model
        print(f"⌛ Loading OpenVINO HiFT model from {model_path}...")
        model = core.read_model(model_path)
        if self.hift_input_len > 0:
            model.reshape([1, 80, self.hift_input_len])
        self.hift = core.compile_model(model, device)
        print(f"✅ HiFT model loaded")

        # ISTFT parameters (matching HiFTGenerator defaults)
        self.istft_params = {"n_fft": 16, "hop_len": 4}
        self.audio_limit = 0.99
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

    def inference(self, speech_feat, finalize=True):
        """
        Run HiFT inference using OpenVINO.

        Args:
            speech_feat: Mel spectrogram (batch, 80, mel_len)
            finalize: Whether this is the final chunk (not used in OV version)

        Returns:
            speech: Generated waveform (batch, samples)
            None: Placeholder for source
        """
        # Prepare input - convert to numpy
        if isinstance(speech_feat, torch.Tensor):
            mel_input = speech_feat.cpu().numpy()
        else:
            mel_input = speech_feat

        # Pad mel_input to fixed length on the third dimension for NPU optimization
        if self.hift_input_len > 0:
            target_len = self.hift_input_len
            original_len = mel_input.shape[2]
            if original_len < target_len:
                # Pad with zeros on the right
                pad_len = target_len - original_len
                mel_input = np.pad(mel_input, ((0, 0), (0, 0), (0, pad_len)), mode="constant", constant_values=0)
            else:
                mel_input = mel_input[:, :, :target_len]
                original_len = target_len

        # Run OpenVINO inference - output is (batch, n_fft+2, time)

        result = self.hift(mel_input)
        x = torch.from_numpy(result[0].copy())

        # Post-processing: exp, sin, istft, clamp (done in Python)
        n_fft = self.istft_params["n_fft"]
        magnitude = torch.exp(x[:, : n_fft // 2 + 1, :])
        phase = torch.sin(x[:, n_fft // 2 + 1 :, :])  # sin is redundancy but kept for compatibility

        speech = self._istft(magnitude, phase)
        speech = torch.clamp(speech, -self.audio_limit, self.audio_limit)

        # Remove padding from output (restore original length)
        # HiFT upsamples mel by hop_size (480), so original_samples = original_len * 480
        if self.hift_input_len > 0:
            if original_len < target_len:
                original_samples = original_len * 480  # hop_size from mel_spectrogram config
                speech = speech[:, :original_samples]
            else:
                speech = speech[:, : original_len * 480]

        return speech, None


class OVCosyVoice3Model:
    """
    OpenVINO-based CosyVoice3 Model for TTS inference.
    Uses OpenVINO models for LLM, Flow and HiFT.
    """

    def __init__(self, llm: OVCosyVoice3LM, flow: OVFlow, hift: OVHiFT, token_mel_ratio: int = 2, pre_lookahead_len: int = 3):
        self.llm = llm  # OVCosyVoice3LM instance
        self.flow = flow  # OVFlow instance
        self.hift = hift  # OVHiFT instance
        # Flow parameters
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
        # NOTE must matching training static_chunk_size
        self.token_hop_len = 25
        # rtf and decoding related
        self.llm_context = nullcontext()  # OpenVINO doesn't need CUDA context
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        """Run LLM inference using OpenVINO model."""
        with self.llm_context:
            # OVCosyVoice3LM.inference() is a generator
            for i in self.llm.inference(
                text=text,
                text_len=torch.tensor([text.shape[1]], dtype=torch.int32),
                prompt_text=prompt_text,
                prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32),
                prompt_speech_token=llm_prompt_speech_token,
                prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32),
                embedding=llm_embedding,
                uuid=uuid,
            ):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def vc_job(self, source_speech_token, uuid):
        self.tts_speech_token_dict[uuid] = source_speech_token.flatten().tolist()
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        """Convert speech tokens to waveform using OpenVINO Flow + HiFT."""
        # Run Flow inference with OpenVINO
        tts_mel, _ = self.flow.inference(
            token=token.to(torch.int32),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32),
            prompt_token=prompt_token,
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32),
            prompt_feat=prompt_feat,
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32),
            embedding=embedding,
            streaming=stream,
            finalize=finalize,
        )

        tts_mel = tts_mel[:, :, token_offset * self.token_mel_ratio :]

        # append mel cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel = self.hift_cache_dict[uuid]["mel"]
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
            self.hift_cache_dict[uuid]["mel"] = tts_mel
        else:
            self.hift_cache_dict[uuid] = {"mel": tts_mel, "speech_offset": 0}

        if speed != 1.0:
            assert token_offset == 0 and finalize is True, "speed change only support non-stream inference mode"
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear")

        # Run HiFT inference with OpenVINO
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
        tts_speech = tts_speech[:, self.hift_cache_dict[uuid]["speech_offset"] :]
        self.hift_cache_dict[uuid]["speech_offset"] += tts_speech.shape[1]

        return tts_speech

    def tts(
        self,
        text=torch.zeros(1, 0, dtype=torch.int32),
        flow_embedding=torch.zeros(0, 192),
        llm_embedding=torch.zeros(0, 192),
        prompt_text=torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat=torch.zeros(1, 0, 80),
        source_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        stream=False,
        speed=1.0,
        **kwargs,
    ):
        """
        Main TTS method - generate speech from text.

        Args:
            text: Input text tokens
            flow_embedding: Speaker embedding for flow
            llm_embedding: Speaker embedding for LLM (not used in OV version)
            prompt_text: Prompt text tokens
            llm_prompt_speech_token: Prompt speech tokens for LLM
            flow_prompt_speech_token: Prompt speech tokens for flow
            prompt_speech_feat: Prompt speech features (mel)
            source_speech_token: Source speech tokens for voice conversion
            stream: Whether to use streaming mode
            speed: Speech speed multiplier

        Yields:
            dict with 'tts_speech' tensor
        """
        import uuid as uuid_module

        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid_module.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        if stream is True:
            token_offset = 0
            prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) * self.token_hop_len - flow_prompt_speech_token.shape[1])
            while True:
                time.sleep(0.1)
                this_token_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= this_token_hop_len + self.pre_lookahead_len:
                    this_tts_speech_token = torch.tensor(
                        self.tts_speech_token_dict[this_uuid][: token_offset + this_token_hop_len + self.pre_lookahead_len]
                    ).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(
                        token=this_tts_speech_token,
                        prompt_token=flow_prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        embedding=flow_embedding,
                        token_offset=token_offset,
                        uuid=this_uuid,
                        stream=stream,
                        finalize=False,
                    )
                    token_offset += this_token_hop_len
                    yield {"tts_speech": this_tts_speech.cpu()}
                if (
                    self.llm_end_dict[this_uuid] is True
                    and len(self.tts_speech_token_dict[this_uuid]) - token_offset < this_token_hop_len + self.pre_lookahead_len
                ):
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                token_offset=token_offset,
                uuid=this_uuid,
                finalize=True,
            )
            yield {"tts_speech": this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                token_offset=0,
                uuid=this_uuid,
                finalize=True,
                speed=speed,
            )
            yield {"tts_speech": this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)


class OVCosyVoice3:
    """
    OpenVINO-based CosyVoice3 TTS system.

    Uses OpenVINO models for all inference components:
    - LLM (text/speech embeddings + transformer)
    - Flow (mel spectrogram generation)
    - HiFT (vocoder for waveform synthesis)

    Usage:
        ov_cosyvoice = OVCosyVoice3(model_dir, ov_model_dir)
        for output in ov_cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav):
            audio = output['tts_speech']
    """

    def __init__(
        self,
        model_dir: str,
        ov_model_dir: str = None,
        device: str = "CPU",
        llm_device: str = None,
        flow_device: str = None,
        hift_device: str = None,
        frontend_device: str = None,
        npu_ov_config: dict = None,
        hift_input_len: int = 0,
    ):
        """
        Initialize OVCosyVoice3.

        Args:
            model_dir: Path to OpenVINO model directory (contains all converted models and dependency files).
                       If dependency files (campplus.onnx, speech_tokenizer_v3.onnx, etc.) are not found here,
                       will look for them in the original model directory.
            ov_model_dir: (Deprecated) If provided, uses this as the OpenVINO model directory.
                          For backward compatibility only.
            device: Default OpenVINO device (CPU, GPU, etc.) for all models
            llm_device: OpenVINO device for LLM model (defaults to device if not specified)
            flow_device: OpenVINO device for Flow model (defaults to device if not specified)
            hift_device: OpenVINO device for HiFT model (defaults to device if not specified)
            frontend_device: OpenVINO device for frontend models (defaults to device if not specified)
        """
        # Handle backward compatibility: if ov_model_dir is provided, use old behavior
        if ov_model_dir is not None:
            self.model_dir = model_dir
            self.ov_model_dir = ov_model_dir
        else:
            # New behavior: model_dir is the OpenVINO model directory with all files
            self.model_dir = model_dir
            self.ov_model_dir = model_dir

        self.ov_device = device

        # Set device for each component (use default device if not specified)
        self.llm_device = llm_device if llm_device is not None else device
        self.flow_device = flow_device if flow_device is not None else device
        self.hift_device = hift_device if hift_device is not None else device
        self.frontend_device = frontend_device if frontend_device is not None else device

        # Check model directory exists
        if not os.path.exists(self.ov_model_dir):
            raise ValueError(f"Model directory not found: {self.ov_model_dir}")

        # Determine where to find config and dependency files
        # First try ov_model_dir, then fall back to model_dir
        config_dir = self.ov_model_dir
        hyper_yaml_path = f"{config_dir}/cosyvoice3.yaml"
        if not os.path.exists(hyper_yaml_path):
            config_dir = self.model_dir
            hyper_yaml_path = f"{config_dir}/cosyvoice3.yaml"
            if not os.path.exists(hyper_yaml_path):
                raise ValueError(f"cosyvoice3.yaml not found in {self.ov_model_dir} or {self.model_dir}!")

        # Determine qwen_pretrain_path
        qwen_path = os.path.join(self.ov_model_dir, "CosyVoice-BlankEN")
        if not os.path.exists(qwen_path):
            qwen_path = os.path.join(self.model_dir, "CosyVoice-BlankEN")

        # Extract config values directly from yaml file using regex (avoid loading PyTorch models)
        # hyperpyyaml's yaml contains special tags that yaml.safe_load cannot parse
        with open(hyper_yaml_path, "r") as f:
            yaml_content = f.read()

        def extract_yaml_value(content, key, default):
            """Extract simple value from yaml content using regex."""
            import re

            pattern = rf"^{key}:\s*(\S+)"
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                value = match.group(1)
                try:
                    return int(value)
                except ValueError:
                    try:
                        return float(value)
                    except ValueError:
                        return value
            return default

        sample_rate = extract_yaml_value(yaml_content, "sample_rate", 24000)
        llm_input_size = extract_yaml_value(yaml_content, "llm_input_size", 896)
        speech_token_size = 6561  # Default value, defined in llm config
        token_mel_ratio = extract_yaml_value(yaml_content, "token_mel_ratio", 2)
        pre_lookahead_len = 3  # Default value

        # Create tokenizer and feat_extractor without loading full hyperpyyaml config
        # This avoids initializing PyTorch models (llm, flow, hift)
        from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
        from matcha.utils.audio import mel_spectrogram
        from functools import partial

        get_tokenizer = partial(get_qwen_tokenizer, token_path=qwen_path, skip_special_tokens=True, version="cosyvoice3")
        feat_extractor = partial(
            mel_spectrogram, n_fft=1920, num_mels=80, sampling_rate=sample_rate, hop_size=480, win_size=1920, fmin=0, fmax=None, center=False
        )
        allowed_special = "all"

        # Determine paths for dependency files (try ov_model_dir first, then model_dir)
        def get_dep_path(filename, required=True):
            ov_path = f"{self.ov_model_dir}/{filename}"
            if os.path.exists(ov_path):
                return ov_path
            model_path = f"{self.model_dir}/{filename}"
            if os.path.exists(model_path):
                return model_path
            if required:
                raise ValueError(f"{filename} not found in {self.ov_model_dir} or {self.model_dir}!")
            return ""  # Return empty string for optional files

        campplus_path = get_dep_path("campplus.onnx")
        speech_tokenizer_path = get_dep_path("speech_tokenizer_v3.onnx")
        spk2info_path = get_dep_path("spk2info.pt", required=False)  # Optional file

        # Initialize OpenVINO frontend (uses OpenVINO for ONNX inference)
        print(f"⌛ Loading OpenVINO frontend models on {self.frontend_device}...")
        self.frontend = OVCosyVoiceFrontEnd(
            get_tokenizer, feat_extractor, campplus_path, speech_tokenizer_path, spk2info_path, allowed_special, device=self.frontend_device
        )
        print(f"✅ OpenVINO frontend loaded on {self.frontend_device}")
        self.sample_rate = sample_rate

        # Load OpenVINO LLM (use pre-extracted config values to avoid PyTorch model initialization)
        print(f"⌛ Loading OpenVINO LLM models on {self.llm_device}...")
        ov_llm = OVCosyVoice3LM(
            model_path=self.ov_model_dir,
            device=self.llm_device,
            speech_token_size=speech_token_size,
            llm_input_size=llm_input_size,
            npu_ov_config=npu_ov_config,
        )
        print(f"✅ OpenVINO LLM loaded on {self.llm_device}")

        # Load OpenVINO Flow and HiFT (use pre-extracted config values)
        print(f"⌛ Loading OpenVINO Flow model on {self.flow_device}...")
        ov_flow = OVFlow(model_dir=self.ov_model_dir, device=self.flow_device, token_mel_ratio=token_mel_ratio, pre_lookahead_len=pre_lookahead_len)
        print(f"✅ OpenVINO Flow loaded on {self.flow_device}")

        print(f"⌛ Loading OpenVINO HiFT model on {self.hift_device}...")
        ov_hift = OVHiFT(model_path=f"{self.ov_model_dir}/{HIFT_PATH}", device=self.hift_device, hift_input_len=hift_input_len)
        print(f"✅ OpenVINO HiFT loaded on {self.hift_device}")

        # Create OVCosyVoice3Model
        self.model = OVCosyVoice3Model(ov_llm, ov_flow, ov_hift, token_mel_ratio, pre_lookahead_len)

    def list_available_spks(self):
        """List available speaker IDs."""
        return list(self.frontend.spk2info.keys())

    def add_zero_shot_spk(self, prompt_text, prompt_wav, zero_shot_spk_id):
        """Add a new zero-shot speaker."""
        assert zero_shot_spk_id != "", "do not use empty zero_shot_spk_id"
        model_input = self.frontend.frontend_zero_shot("", prompt_text, prompt_wav, self.sample_rate, "")
        del model_input["text"]
        del model_input["text_len"]
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        """Save speaker info to file."""
        torch.save(self.frontend.spk2info, f"{self.model_dir}/spk2info.pt")

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        """Supervised fine-tuning inference."""
        from cosyvoice.utils.file_utils import logging

        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                logging.info(f"yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}")
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, zero_shot_spk_id="", stream=False, speed=1.0, text_frontend=True):
        """Zero-shot TTS inference with voice cloning."""
        from cosyvoice.utils.file_utils import logging

        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning(f"synthesis text {i} too short than prompt text {prompt_text}, this may lead to bad performance")
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                logging.info(f"yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}")
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_wav, zero_shot_spk_id="", stream=False, speed=1.0, text_frontend=True):
        """Cross-lingual TTS inference."""
        from cosyvoice.utils.file_utils import logging

        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                logging.info(f"yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}")
                yield model_output
                start_time = time.time()

    def inference_instruct2(self, tts_text, instruct_text, prompt_wav, zero_shot_spk_id="", stream=False, speed=1.0, text_frontend=True):
        """Instruct-based TTS inference."""
        from cosyvoice.utils.file_utils import logging

        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                logging.info(f"yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}")
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_wav, prompt_wav, stream=False, speed=1.0):
        """Voice conversion inference."""
        from cosyvoice.utils.file_utils import logging

        model_input = self.frontend.frontend_vc(source_wav, prompt_wav, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
            speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
            logging.info(f"yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}")
            yield model_output
            start_time = time.time()
