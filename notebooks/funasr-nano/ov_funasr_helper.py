import openvino as ov
import nncf
from pathlib import Path
import torch
import logging
import random
import re
import string
import time
import traceback
import types
from typing import Optional, Tuple, Callable, Any
import openvino.opset13 as opset13
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
import numpy as np
import gc
from transformers.utils import is_torch_xpu_available
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
import shutil
import json
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from dataclasses import dataclass
from transformers.cache_utils import DynamicCache, DynamicLayer
from funasr.utils.datadir_writer import DatadirWriter
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


def patched_dynamic_layer_update(
    self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: dict[str, Any] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if self.keys is None:
        self.keys = key_states
        self.values = value_states
        self.device = key_states.device
        self.dtype = key_states.dtype
        self.is_initialized = True
    else:
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values


DynamicLayer.update = patched_dynamic_layer_update


def dynamic_cache_from_legacy(past_key_values):
    """Build a ``DynamicCache`` from a legacy tuple-of-tuples cache.

    ``DynamicCache.from_legacy_cache`` was removed in recent ``transformers``
    releases, so reconstruct the cache through the public ``update`` API.
    """
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(past_key_values)
    cache = DynamicCache()
    for layer_idx, (key_states, value_states) in enumerate(past_key_values):
        cache.update(key_states, value_states, layer_idx)
    return cache


def dynamic_cache_to_legacy(cache):
    """Convert a ``DynamicCache`` back to the legacy tuple-of-tuples format."""
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple((layer.keys, layer.values) for layer in cache.layers)


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


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


def prepare_padding_mask(attention_mask: Optional[torch.Tensor], kv_length: int, kv_offset: int, _slice: bool = True) -> Optional[torch.Tensor]:
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it, and slicing
    according to the `kv_offset` if `_slice` is `True`.
    """
    local_padding_mask = attention_mask
    if attention_mask is not None:
        # Pad it if necessary
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
        # For flex, we should not slice them, only use an offset
        if _slice:
            # Equivalent to: `local_padding_mask = attention_mask[:, kv_offset : kv_offset + kv_length]`,
            # but without data-dependent slicing (i.e. torch.compile friendly)
            mask_indices = torch.arange(kv_length, device=local_padding_mask.device)
            mask_indices += kv_offset
            local_padding_mask = local_padding_mask[:, mask_indices]
    return local_padding_mask


def and_masks(*mask_functions: list[Callable]) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask


def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Note that here the mask should ALWAYS be at least of the max `kv_index` size in the dimension 1. This is because
        # we cannot pad it here in the mask_function as we don't know the final size, and we cannot try/except, as it is not
        # vectorizable on accelerator devices
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def _ignore_causal_mask_sdpa(
    padding_mask: Optional[torch.Tensor],
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: Optional[int] = None,
) -> bool:
    """
    Detects whether the causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the 2D `padding_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    is_tracing = torch.jit.is_tracing() or isinstance(padding_mask, torch.fx.Proxy) or is_torchdynamo_compiling()
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = torch.arange(kv_length, device=padding_mask.device)
        mask_indices += kv_offset
        padding_mask = padding_mask[:, mask_indices]

    # When using `torch.export` or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is
    # hard-coded to the forward. If a user exports a model with query_length > 1, the exported model will hard-code `is_causal=True`
    # which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108). Thus, we only set
    # `ignore_causal_mask = True` if we are not tracing
    if (
        not is_tracing
        # only cases when lower and upper diags are the same, see https://github.com/pytorch/pytorch/issues/108108
        and (query_length == 1 or (kv_length == query_length or is_torch_xpu_available))
        # in this case we need to add special patterns to the mask so cannot be skipped otherwise
        and (local_attention_size is None or kv_length < local_attention_size)
        # In this case, we need to add padding to the mask, so cannot be skipped otherwise
        and (padding_mask is None or (padding_mask.all() if not is_torch_xpu_available or query_length == 1 else padding_mask[:, :query_length].all()))
    ):
        return True

    return False


def sdpa_mask_without_vmap(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Optional[Callable] = None,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    if mask_function is None:
        mask_function = causal_mask_function

    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    # Create broadcatable indices
    device = cache_position.device
    q_indices = cache_position[None, None, :, None]
    head_indices = torch.arange(1, dtype=torch.long, device=device)[None, :, None, None]
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[:, None, None, None]
    kv_indices = torch.arange(kv_length, dtype=torch.long, device=device)[None, None, None, :] + kv_offset

    # Apply mask function element-wise through broadcasting
    causal_mask = mask_function(batch_indices, head_indices, q_indices, kv_indices)
    # Expand the mask to match batch size and query length if they weren't used in the mask function
    causal_mask = causal_mask.expand(batch_size, -1, q_length, kv_length)

    return causal_mask


# Adapted from https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/masking_utils.py#L433
# Specifically for OpenVINO, we use torch.finfo(torch.float16).min instead of torch.finfo(dtype).min
def eager_mask_without_vmap(*args, **kwargs) -> Optional[torch.Tensor]:
    kwargs.pop("allow_is_causal_skip", None)
    dtype = kwargs.get("dtype", torch.float32)
    mask = sdpa_mask_without_vmap(*args, allow_is_causal_skip=False, **kwargs)
    # we use torch.finfo(torch.float16).min instead torch.finfo(dtype).min to avoid an overflow but not
    # sure this is the right way to handle this, we are basically pretending that -65,504 is -inf
    mask = torch.where(
        mask,
        torch.tensor(0.0, device=mask.device, dtype=dtype),
        torch.tensor(torch.finfo(torch.float16).min, device=mask.device, dtype=dtype),
    )
    return mask


# for OpenVINO, we use torch.finfo(torch.float16).min instead of torch.finfo(dtype).min
# Although I'm not sure this is the right way to handle this, we are basically pretending that -65,504 is -inf
ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask_without_vmap)

# for decoder models, we use eager mask without vmap for sdpa as well
# to avoid a nan output issue in OpenVINO that only happens in case of:
# non-stateful models on cpu and stateful models on npu
ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", eager_mask_without_vmap)


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


core = ov.Core()


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


TEXT_EMBEDDINGS_PATH = "openvino_text_embeddings_model.xml"
ENCODER_PATH = "openvino_encoder_model.xml"
LANGUAGE_PATH = "openvino_model.xml"
FRONTEND_CONFIG_PATH = "frontend_config.json"


def convert_funasr(model_id, model_path=None, quantization_config=None):

    if model_path is None:
        model_path = Path(model_id.split("/")[-1])
    else:
        model_path = Path(model_path)

    if all((model_path / model_name).exists() for model_name in [FRONTEND_CONFIG_PATH, TEXT_EMBEDDINGS_PATH, ENCODER_PATH, LANGUAGE_PATH]):
        print(f"✅ {model_id} model already converted. You can find results in {model_path}")
        return model_path
    print(f"⌛ {model_id} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")
    from model import FunASRNano

    pt_model, kwargs = FunASRNano.from_pretrained(model=model_id, device="cpu")
    kwargs
    pt_model = pt_model.to(torch.float32)
    print("✅ Original model successfully loaded")
    print("⌛ Export tokenizer and config")
    kwargs["tokenizer"].save_pretrained(model_path)
    for json_file in Path(model_id + "/Qwen3-0.6B").glob("*.json"):
        shutil.copy(json_file, model_path / json_file.name)

    # Export frontend config
    if kwargs.get("frontend") is not None:
        frontend = kwargs["frontend"]
        frontend_config = {
            # Frontend settings
            "frontend_type": "WavFrontend",
            "fs": frontend.fs,
            "frame_shift": frontend.frame_shift,
            "frame_length": frontend.frame_length,
            "lfr_m": frontend.lfr_m,
            "lfr_n": frontend.lfr_n,
            "n_mels": frontend.n_mels,
            "window": frontend.window,
            "dither": 0.0,  # Set to 0 for deterministic inference (original uses dither=1.0 which adds random noise)
            # Inference kwargs for data_load_speech
            "dataset_conf": kwargs.get("dataset_conf", {}),
            "multiturn_num_max": kwargs.get("multiturn_num_max", 5),
            "max_token_length": kwargs.get("max_token_length", 1500),
            "infer_with_assistant_input": kwargs.get("infer_with_assistant_input", False),
            "data_type": kwargs.get("data_type", "sound"),
            "max_length": kwargs.get("max_length", 512),
            "batch_size": kwargs.get("batch_size", 1),
        }
        with open(model_path / FRONTEND_CONFIG_PATH, "w") as f:
            json.dump(frontend_config, f, indent=2)
        print("✅ Frontend config exported")

    if not (model_path / TEXT_EMBEDDINGS_PATH).exists():
        print("⌛ Convert TEXT_EMBEDDINGS model")

        ov_model = ov.convert_model(pt_model.llm.model.get_input_embeddings(), example_input=torch.ones([1, 35], dtype=torch.int32))
        ov.save_model(ov_model, model_path / TEXT_EMBEDDINGS_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ TEXT_EMBEDDINGS model successfully converted")

    if not (model_path / ENCODER_PATH).exists():
        print("⌛ Convert ENCODER_PATH model")

        def forward_wrap_encoder(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
            encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
            return encoder_out, encoder_out_lens

        pt_model._orig_forward = pt_model.forward
        pt_model.forward = types.MethodType(forward_wrap_encoder, pt_model)
        example_input = {
            "speech": torch.ones([1, 94, 560], dtype=torch.float32),
            "speech_lengths": torch.tensor([1]).to(dtype=torch.int32),
        }

        ov_model = ov.convert_model(pt_model, example_input=example_input)
        ov.save_model(ov_model, model_path / ENCODER_PATH)
        del ov_model
        pt_model.forward = pt_model._orig_forward
        del pt_model._orig_forward
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ ENCODER model successfully converted")

    if not (model_path / LANGUAGE_PATH).exists():
        print("⌛ Convert LANGUAGE_MODEL model")
        patch_cos_sin_cached_fp32(pt_model.llm)
        if hasattr(pt_model.llm, "model"):
            patch_cos_sin_cached_fp32(pt_model.llm.model)

        def forward_wrap(
            self,
            attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
        ):
            if past_key_values is not None:
                pkv = dynamic_cache_from_legacy(past_key_values)
            outputs = self._orig_forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=pkv,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
            return (outputs.logits, dynamic_cache_to_legacy(outputs.past_key_values))

        num_pkv = pt_model.llm.config.num_hidden_layers
        hidden_size = pt_model.llm.config.hidden_size

        pt_model.llm._orig_forward = pt_model.llm.forward
        pt_model.llm.forward = types.MethodType(forward_wrap, pt_model.llm)

        num_pkv = pt_model.llm.config.num_hidden_layers
        hidden_size = pt_model.llm.config.hidden_size

        pkv_shape = (
            2,
            pt_model.llm.config.num_key_value_heads,
            2,
            pt_model.llm.config.head_dim,
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
                        pt_model.llm.config.num_key_value_heads,
                        -1,
                        pt_model.llm.config.head_dim,
                    ]
                )
            ]
            * 2
            * num_pkv
        )
        input_shapes += [ov.PartialShape([-1, -1, hidden_size])]  # inputs_embeds
        __make_16bit_traceable(pt_model.llm)

        ov_model = ov.convert_model(pt_model.llm, example_input=example_input, input=input_shapes)
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
        ov_model.set_rt_info("8.0", ["runtime_options", "ACTIVATIONS_SCALE_FACTOR"])
        ov.save_model(ov_model, model_path / LANGUAGE_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()

    del pt_model
    gc.collect()
    print(f"✅ {model_id} model conversion finished. You can find results in {model_path}")
    return model_path


@dataclass
class Segment:
    speaker: str
    text: str
    audio: torch.Tensor


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    decoder_loss_weight: float
    use_text_loss: bool


def to_device(x, device):
    """Send tensor or dict of tensors to device.

    Args:
        x (Tensor or dict): Torch tensor or dict of tensors.
        device (str or torch.device): Target device.

    Returns:
        Tensor or dict: Torch tensor(s) on the target device.
    """
    if isinstance(x, dict):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def load_frontend_from_config(config_path):
    """
    Load frontend from saved config file.

    Args:
        config_path: Path to frontend_config.json

    Returns:
        frontend object
    """
    from funasr.register import tables

    with open(config_path, "r") as f:
        frontend_config = json.load(f)

    frontend_type = frontend_config.pop("frontend_type", "WavFrontend")
    frontend_class = tables.frontend_classes.get(frontend_type)
    frontend = frontend_class(**frontend_config)
    return frontend


class OVModelForCausalLMWithEmbed(OVModelForCausalLM):
    """
    Wrapper for OVModelForCausalLM that supports inputs_embeds input.
    This is needed for multimodal models where we need to pass pre-computed embeddings
    (e.g., audio embeddings merged with text embeddings).
    """

    def set_token_emb(self, token_emb_path):
        """Set the token embedding model path after from_pretrained."""
        self.token_emb = core.read_model(token_emb_path)
        self.token_emb_request = None

    def _compile_token_emb(self):
        if self.token_emb_request is None:
            self.token_emb_request = core.compile_model(self.token_emb, "CPU" if self._device == "NPU" else self._device)

    def to(self, device: str):
        if isinstance(device, str):
            self._device = device.upper()
            self.clear_requests()

        return self

    def clear_requests(self):
        del self.request
        del self.token_emb_request
        self.request = None
        self.token_emb_request = None

    def embed_tokens(self, input_ids: torch.LongTensor):
        self._compile_token_emb()
        res = self.token_emb_request(input_ids, share_inputs=True)
        return res[0]

    def prepare_inputs(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> dict:
        # Determine batch_size from inputs_embeds or input_ids
        if inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            seq_length = inputs_embeds.shape[1]
        else:
            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]

        inputs = {}

        # Handle stateful model state reset
        if self.stateful:
            if past_key_values is None:
                if self.request is not None:
                    self.request.reset_state()
                self.next_beam_idx = np.arange(batch_size, dtype=int)
                self._past_length = 0

        past_len = self._get_past_length(past_key_values)

        # Use inputs_embeds if provided, otherwise use input_ids
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids if past_key_values is None else input_ids[:, -1:])

            if hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb
        inputs["inputs_embeds"] = inputs_embeds

        # Handle attention_mask
        if "attention_mask" in self.input_names or "position_ids" in self.input_names:
            if attention_mask is not None:
                attention_mask_np = attention_mask.cpu().numpy() if isinstance(attention_mask, torch.Tensor) else attention_mask
            else:
                attention_mask_np = np.ones((batch_size, seq_length + past_len), dtype=np.int64)

            if "attention_mask" in self.input_names:
                inputs["attention_mask"] = attention_mask_np

        # Handle position_ids
        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids_np = position_ids.cpu().numpy() if isinstance(position_ids, torch.Tensor) else position_ids
            else:
                position_ids_np = np.cumsum(attention_mask_np, axis=1) - 1
                position_ids_np[attention_mask_np == 0] = 1
                if past_key_values:
                    position_ids_np = position_ids_np[:, -seq_length:]
            inputs["position_ids"] = position_ids_np

        # Handle beam_idx for beam search
        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)
        return inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        self.compile()

        inputs = self.prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).clone().to(self.device)

        # Determine sequence length for past_length update
        if inputs_embeds is not None:
            seq_length = inputs_embeds.shape[1]
        else:
            seq_length = input_ids.shape[1]

        if self.stateful:
            past_key_values = ((),)
            self._past_length += seq_length
        else:
            if self.use_cache:
                past_key_values = tuple(np.copy(self.request.get_tensor(key).data) for key in self.key_value_output_names)
                past_key_values = tuple(past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv))
            else:
                past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        # Only use inputs_embeds for the first forward pass
        if past_key_values is not None:
            past_len = self._get_past_length(past_key_values)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_len) :]
            elif past_len < input_ids.shape[1]:
                input_ids = input_ids[:, past_len:]
            # After first pass, don't use inputs_embeds
            inputs_embeds = None

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None and "position_ids" in self.input_names:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
        }

        return model_inputs


class OVFunASRNano:
    def __init__(self, pretrained_dir, device, llm_ov_config={}):
        self.device = device
        # Map OpenVINO device to torch device for tensor operations
        self.torch_device = "cpu"  # OpenVINO CPU/GPU models use CPU tensors for input
        self.feat_permute = True

        model_dir = Path(pretrained_dir)
        # Use OVModelForCausalLMWithEmbed to support inputs_embeds for multimodal fusion
        self.llm = OVModelForCausalLMWithEmbed.from_pretrained(model_dir, device=self.device, ov_config=llm_ov_config)

        self.llm.set_token_emb(model_dir / TEXT_EMBEDDINGS_PATH)
        # Disable Snippets optimization to avoid internal error with certain model structures
        encoder_config = {"SNIPPETS_MODE": "DISABLE"} if self.device == "CPU" else {}
        self.audio_encoder = core.compile_model(model_dir / ENCODER_PATH, self.device if self.device != "NPU" else "CPU", encoder_config)

        # Load tokenizer from saved config
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"✅ Tokenizer loaded from {model_dir}")

        # Load frontend from saved config
        frontend_config_path = model_dir / FRONTEND_CONFIG_PATH
        if frontend_config_path.exists():
            self.frontend = load_frontend_from_config(frontend_config_path)
            # Load inference kwargs from the same config
            with open(frontend_config_path, "r") as f:
                config = json.load(f)
            self.inference_kwargs = {
                "dataset_conf": config.get("dataset_conf", {}),
                "multiturn_num_max": config.get("multiturn_num_max", 5),
                "max_token_length": config.get("max_token_length", 1500),
                "infer_with_assistant_input": config.get("infer_with_assistant_input", False),
                "data_type": config.get("data_type", "sound"),
                "max_length": config.get("max_length", 512),
                "batch_size": config.get("batch_size", 1),
            }
            print(f"✅ Frontend and inference config loaded from {frontend_config_path}")
        else:
            self.frontend = None
            self.inference_kwargs = {}
            print(f"⚠️ Frontend config not found at {frontend_config_path}, frontend will need to be provided manually")

    def data_template(self, data):
        system, user, assistant = [], [], []
        for i, item in enumerate(data):
            role = item["role"]
            content = item["content"]
            if role == "system":
                system.append(content)
            elif role == "user":
                if "audio" in item:
                    audio = item["audio"]
                    content = [content, audio]
                user.append(content)
            elif role == "assistant":
                assistant.append(content)

        system = system * len(user)

        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
        }

        return contents

    def data_load_speech(self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs):
        system = contents["system"]
        user = contents["user"]
        assistant = contents["assistant"]
        pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        do_think = True
        sys_prompt = True
        if "dataset_conf" in kwargs:
            do_think = kwargs["dataset_conf"].get("do_think", True)
            sys_prompt = kwargs["dataset_conf"].get("sys_prompt", True)

        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        input_source_ids = []
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            if len(input_ids) > kwargs.get("max_token_length", 1500):
                break
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            if i == 0:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}"
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            if not do_think:
                source_input += "<think>\n\n</think>\n\n"

            splits = pattern.split(source_input)
            source_ids = []
            fbank_mask_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
            speech, speech_lengths = [], []
            for k, sub_str in enumerate(splits):
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                    fbank_mask_i += [0] * len(sub_token)
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace("<|endofspeech|>", "")
                    if sub_str.startswith("!"):
                        sub_str = sub_str[1:]
                        if sub_str.startswith("!"):  # !!: audio sample point
                            sub_str = audio
                        try:
                            time1 = time.perf_counter()
                            data_src = load_audio_text_image_video(sub_str, fs=frontend.fs, **kwargs)
                            time2 = time.perf_counter()
                            meta_data["load_data"] = f"{time2 - time1:0.3f}"
                        except Exception as e:
                            logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")

                        speech, speech_lengths = extract_fbank(
                            data_src,
                            data_type=kwargs.get("data_type", "sound"),
                            frontend=frontend,
                            is_final=True,
                        )  # speech: [b, T, d]

                        time3 = time.perf_counter()
                        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
                        meta_data["batch_data_time"] = speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000

                        if self.feat_permute:
                            speech = speech.permute(0, 2, 1)

                        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                        olens = 1 + (olens - 3 + 2 * 1) // 2
                        fake_token_len_i = (olens - 1) // 2 + 1
                        fake_token = [0] * fake_token_len_i
                        fbank_beg_i = len(source_ids)
                        source_ids += fake_token
                        fbank_mask_i += [1] * len(fake_token)

            fbank_beg += [fbank_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            source_mask = [-100] * len(source_ids)
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            fbank_mask += fbank_mask_i
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

        fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
        fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
        fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        if len(fbank) > 0:
            speech = torch.nn.utils.rnn.pad_sequence(fbank, batch_first=True, padding_value=0.0)
            speech_lengths = torch.nn.utils.rnn.pad_sequence(fbank_lens, batch_first=True, padding_value=-1)
        else:
            speech = []
            speech_lengths = []
        output = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "fbank_mask": fbank_mask[None, :],
            "fbank_beg": fbank_beg[None,],
            "fake_token_len": fake_token_len[None, :],
            "input_ids": input_ids[None,],
            "attention_mask": attention_mask[None,],
            "labels_ids": labels,
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }

        return output

    def encode(self, speech, speech_lengths):
        # audio encoder
        if self.feat_permute:
            speech_permuted = speech.permute(0, 2, 1)

            output = self.audio_encoder([speech_permuted, speech_lengths])
            encoder_out, encoder_out_lens = torch.from_numpy(output[0]), torch.from_numpy(output[1])
        else:

            output = self.audio_encoder([speech, speech_lengths])
            encoder_out, encoder_out_lens = torch.from_numpy(output[0]), torch.from_numpy(output[1])
        return encoder_out, encoder_out_lens

    def inference_prepare(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        meta_data = {}

        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        output = self.data_load_speech(contents, tokenizer, frontend, meta_data=meta_data, **kwargs)
        batch = to_device(output, self.torch_device)

        # audio encoder
        speech = batch["speech"]

        if len(speech) > 0:
            if "audio_embedding" in kwargs and "audio_embedding_lens" in kwargs:
                encoder_out = kwargs["audio_embedding"]
                encoder_out_lens = kwargs["audio_embedding_lens"]
            else:
                speech_lengths = batch["speech_lengths"][:, 0]
                # fp16
                if kwargs.get("fp16", False):
                    speech = speech.to(torch.float16)
                elif kwargs.get("bf16", False):
                    speech = speech.to(torch.bfloat16)
                # audio encoder

                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
                meta_data["audio_adaptor_out"] = encoder_out
                meta_data["audio_adaptor_out_lens"] = encoder_out_lens

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        fbank_beg = batch["fbank_beg"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("tearchforing", False):
            input_ids = source_ids

        input_ids[input_ids < 0] = 0
        inputs_embeds = torch.from_numpy(self.llm.embed_tokens(input_ids))
        batch_size, token_num, dims = inputs_embeds.shape

        fake_token_len[fake_token_len < 0] = 0
        fbank_beg[fbank_beg < 0] = 0

        speech_idx = 0
        for batch_idx in range(batch_size):
            for turn_id in range(fbank_beg.shape[1]):
                fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                if fbank_beg_idx > 0:
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    speech_token = encoder_out[speech_idx, :speech_token_len, :]

                    try:
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token
                    except Exception as e:
                        #
                        logging.error(f"{str(e)}, {traceback.format_exc()}")
                        logging.info(
                            f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                        )
                        speech_token_len = encoder_out_lens[speech_idx].item()
                        speech_token = encoder_out[speech_idx, :speech_token_len, :]
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token

                    speech_idx += 1
        return inputs_embeds, contents, batch, source_ids, meta_data

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        # Use class frontend if not provided
        if frontend is None:
            frontend = self.frontend
        if frontend is None:
            raise ValueError("frontend is required but not provided and not loaded from config")

        # Use class tokenizer if not provided
        if tokenizer is None:
            tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("tokenizer is required but not provided and not loaded from config")

        # Merge saved inference_kwargs with provided kwargs (provided kwargs take precedence)
        merged_kwargs = {**self.inference_kwargs, **kwargs}

        new_data_in = []
        for data in data_in:
            if isinstance(data, str):
                new_data_in.append(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": f"语音转写：<|startofspeech|>!{data}<|endofspeech|>",
                        },
                        {"role": "assistant", "content": "null"},
                    ]
                )
            elif isinstance(data, torch.Tensor):
                new_data_in.append(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": f"语音转写：<|startofspeech|>!!<|endofspeech|>",
                            "audio": data,
                        },
                        {"role": "assistant", "content": "null"},
                    ]
                )
        data_in = new_data_in

        if key is None:
            key = []
            for _ in data_in:
                chars = string.ascii_letters + string.digits
                key.append("rand_key_" + "".join(random.choice(chars) for _ in range(13)))  # nosec B311 - non-secret inference key

        return self.inference_llm(
            data_in,
            data_lengths=data_lengths,
            key=key,
            tokenizer=tokenizer,
            frontend=frontend,
            **merged_kwargs,
        )

    def inference_llm(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(data_in, data_lengths, key, tokenizer, frontend, **kwargs)
        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        with torch.no_grad():
            label = contents["assistant"][-1]
            llm_kwargs = kwargs.get("llm_kwargs", {})
            if not kwargs.get("teachforing", False):
                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=kwargs.get("max_length", 512),
                    **llm_kwargs,
                )

                response = tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]

                loss = None
            else:
                labels_ids = batch["labels_ids"]
                labels_ids[labels_ids == -1] = -100
                attention_mask = batch.get("attention_mask", None)
                model_outputs = self.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels_ids,
                    **llm_kwargs,
                )

                preds = torch.argmax(model_outputs.logits, -1)[:, source_ids.shape[1] :]
                response = tokenizer.batch_decode(
                    preds,
                    add_special_tokens=False,
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]
                loss = model_outputs.loss.item()

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{0 + 1}best_recog"]

        results = []
        response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        result_i = {
            "key": key[0],
            "text": re.sub(r"\s+", " ", response.replace("/sil", " ")),
            "text_tn": response_clean,
            "label": label,
        }
        if loss is not None:
            result_i["loss"] = loss
        results.append(result_i)

        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = response.replace("\n", " ")
            ibest_writer["label"][key[0]] = label.replace("\n", " ")
            ibest_writer["text_tn"][key[0]] = response_clean

        return results, meta_data
