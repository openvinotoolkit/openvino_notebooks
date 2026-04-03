import gc
import sys
import types
from pathlib import Path
from typing import Any, Optional

import numpy as np
import openvino as ov
from typing import Optional, Tuple, Callable, Any
from dataclasses import dataclass

# Import torch and transformers only when needed for conversion
try:
    import torch
    from huggingface_hub import snapshot_download
    from torch import nn
    from transformers.cache_utils import DynamicCache
    from transformers.generation import GenerationMixin, GenerationConfig
    from transformers.modeling_outputs import ModelOutput

    # Try to import DynamicLayer
    try:
        from transformers.cache_utils import DynamicLayer

        DYNAMIC_LAYER_AVAILABLE = True
    except ImportError:
        DynamicLayer = None
        DYNAMIC_LAYER_AVAILABLE = False
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    DYNAMIC_LAYER_AVAILABLE = False
    DynamicLayer = None

# Try to import masking_utils (for patching torch.diff)
try:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    MASKING_AVAILABLE = True
except ImportError:
    ALL_MASK_ATTENTION_FUNCTIONS = {}
    MASKING_AVAILABLE = False

# Import nncf only when needed for compression
try:
    import nncf

    NNCF_AVAILABLE = True
except ImportError:
    NNCF_AVAILABLE = False

try:
    from openvino import opset13
except ImportError:
    from openvino.runtime import opset13

from openvino.frontend.pytorch.patch_model import __make_16bit_traceable

# Add path for Qwen3-TTS module
sys.path.insert(0, str(Path(__file__).parent / "Qwen3-TTS"))


def patch_torch_diff_for_openvino():
    """
    Patch torch.diff to use a version compatible with OpenVINO conversion.
    torch.diff is not supported in OpenVINO, so we replace it with equivalent operations.
    """
    if not TORCH_AVAILABLE or not MASKING_AVAILABLE:
        return

    try:
        import transformers.masking_utils as masking_utils

        # Save original function
        original_find_packed = masking_utils.find_packed_sequence_indices

        def patched_find_packed_sequence_indices(position_ids):
            """
            OpenVINO-compatible version of find_packed_sequence_indices.
            Replaces torch.diff with manual difference calculation.
            """
            # Original code:
            # first_dummy_value = position_ids[:, :1] - 1
            # position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
            # packed_sequence_mask = (position_diff != 1).cumsum(-1)

            # OpenVINO-compatible replacement:
            # torch.diff(x, prepend=y, dim=-1) is equivalent to:
            # torch.cat([y, x], dim=-1)[:, 1:] - torch.cat([y, x], dim=-1)[:, :-1]

            first_val = position_ids[:, :1] - 1
            # Manually compute diff: [first_val, pos[0], pos[1], ...] -> diff
            prepended = torch.cat([first_val, position_ids], dim=-1)
            position_diff = prepended[:, 1:] - prepended[:, :-1]

            packed_sequence_mask = (position_diff != 1).cumsum(-1)
            return packed_sequence_mask

        # Replace the function
        masking_utils.find_packed_sequence_indices = patched_find_packed_sequence_indices
        print("✅ Patched torch.diff in masking_utils.find_packed_sequence_indices for OpenVINO compatibility")
    except Exception as e:
        print(f"⚠️ Could not patch masking_utils: {e}")


# Apply patch before importing Qwen3-TTS
if TORCH_AVAILABLE and MASKING_AVAILABLE:
    patch_torch_diff_for_openvino()

# Import Qwen3-TTS only when torch is available (for conversion)
# Use lazy import to avoid loading unnecessary dependencies
if TORCH_AVAILABLE:
    # Don't import the full qwen_tts package to avoid unnecessary dependencies
    # We'll import specific modules as needed in each conversion function
    pass


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


if TORCH_AVAILABLE:
    DynamicLayer.update = patched_dynamic_layer_update


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
if TORCH_AVAILABLE:
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
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    gather_dim: int,
):
    """
    Fuses reordered cache during generate cycle into ov.Model.
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
    """
    Build initialization ShapeOf Expression for all ReadValue ops
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
    """
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


def patch_stateful(ov_model, dim):
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


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


core = ov.Core()

# File naming conventions for Qwen3-TTS
TALKER_LANGUAGE_NAME = "openvino_talker_language_model.xml"
TALKER_EMBEDDING_NAME = "openvino_talker_embedding_model.xml"
TALKER_TEXT_EMBEDDING_NAME = "openvino_talker_text_embedding_model.xml"
TALKER_TEXT_PROJECTION_NAME = "openvino_talker_text_projection_model.xml"

TALKER_CODE_PREDICTOR_EMBEDDING_NAME = "openvino_talker_code_predictor_embedding_model.xml"
TALKER_CODE_PREDICTOR_NAME = "openvino_talker_code_predictor_model.xml"

SPEAKER_ENCODER_NAME = "openvino_speaker_encoder_model.xml"

# Speech tokenizer model names
SPEECH_TOKENIZER_ENCODER_NAME = "openvino_speech_tokenizer_encoder_model.xml"
SPEECH_TOKENIZER_DECODER_NAME = "openvino_speech_tokenizer_decoder_model.xml"


def convert_qwen3_tts_model(model_id, output_dir, quantization_config=None, use_local_dir=False):
    """
    Convert Qwen3-TTS model to OpenVINO format.

    Args:
        model_id: HuggingFace model ID or local path
        output_dir: Directory to save the converted models
        quantization_config: Optional quantization configuration for nncf
        use_local_dir: Whether to download to local directory first
    """
    # Import here to avoid loading unnecessary dependencies at module import time
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models import Qwen3TTSProcessor

    output_dir = Path(output_dir)

    talker_lang_path = output_dir / TALKER_LANGUAGE_NAME
    talker_embedding_path = output_dir / TALKER_EMBEDDING_NAME
    talker_text_embedding_path = output_dir / TALKER_TEXT_EMBEDDING_NAME
    talker_text_projection_path = output_dir / TALKER_TEXT_PROJECTION_NAME
    talker_code_predictor_embedding_path = output_dir / TALKER_CODE_PREDICTOR_EMBEDDING_NAME
    talker_code_predictor_path = output_dir / TALKER_CODE_PREDICTOR_NAME
    speaker_encoder_path = output_dir / SPEAKER_ENCODER_NAME

    if all(
        [
            talker_lang_path.exists(),
            talker_embedding_path.exists(),
            talker_text_embedding_path.exists(),
            talker_text_projection_path.exists(),
            talker_code_predictor_embedding_path.exists(),
            talker_code_predictor_path.exists(),
        ]
    ):
        print(f"✅ {model_id} model already converted. You can find results in {output_dir}")
        return

    print(f"⌛ {model_id} conversion started. Be patient, it may take some time.")
    print("⌛ Load Original model")

    if use_local_dir:
        ckpt = Path(output_dir) / "ckpt"
        if not ckpt.exists():
            snapshot_download(model_id, local_dir=ckpt, force_download=True)
    else:
        ckpt = model_id

    config = Qwen3TTSConfig.from_pretrained(ckpt)
    config.talker_config._attn_implementation_autoset = False
    config.talker_config._attn_implementation = "sdpa"
    config.talker_config.code_predictor_config._attn_implementation_autoset = False
    config.talker_config.code_predictor_config._attn_implementation = "sdpa"

    model = Qwen3TTSForConditionalGeneration.from_pretrained(ckpt, config=config, torch_dtype=torch.float16)
    model.eval()
    processor = Qwen3TTSProcessor.from_pretrained(ckpt)

    config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Clean up config.json after saving to remove model_type from speaker_encoder_config
    # PretrainedConfig.to_dict() automatically adds model_type, but Qwen3TTSSpeakerEncoderConfig.__init__()
    # doesn't accept it, causing errors during loading. We need to remove it from the saved JSON file.
    import json

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config_json = json.load(f)
        if "speaker_encoder_config" in config_json and "model_type" in config_json["speaker_encoder_config"]:
            del config_json["speaker_encoder_config"]["model_type"]
            with open(config_path, "w") as f:
                json.dump(config_json, f, indent=2)
            print("✅ Cleaned up config.json (removed model_type from speaker_encoder_config)")

    print("✅ Original model successfully loaded")

    # Convert talker embedding model (codec embedding)
    if not talker_embedding_path.exists():
        print("⌛ Convert talker embedding model")
        __make_16bit_traceable(model.talker.get_input_embeddings())
        ov_model = ov.convert_model(
            model.talker.get_input_embeddings(),
            example_input=torch.ones([2, 2], dtype=torch.int64),
        )
        ov.save_model(ov_model, talker_embedding_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Talker embedding model successfully converted")

    # Convert talker text embedding model
    if not talker_text_embedding_path.exists():
        print("⌛ Convert talker text embedding model")
        __make_16bit_traceable(model.talker.get_text_embeddings())
        ov_model = ov.convert_model(
            model.talker.get_text_embeddings(),
            example_input=torch.ones([2, 2], dtype=torch.int64),
        )
        ov.save_model(ov_model, talker_text_embedding_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Talker text embedding model successfully converted")

    # Convert talker text_projection model
    if not talker_text_projection_path.exists():
        print("⌛ Convert talker text_projection model")
        __make_16bit_traceable(model.talker.text_projection)
        text_hidden_size = config.talker_config.text_hidden_size
        ov_model = ov.convert_model(
            model.talker.text_projection,
            example_input=torch.ones([1, 3, text_hidden_size], dtype=torch.float32),
            input=[ov.PartialShape([1, -1, text_hidden_size])],
        )
        ov.save_model(ov_model, talker_text_projection_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Talker text_projection model successfully converted")

    # Convert Talker Language model
    if not talker_lang_path.exists():
        print("⌛ Convert Talker Language model")

        def forward_wrap_talker(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            if past_key_values is not None:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            outputs = self.model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
            if past_key_values is not None:
                outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()

            hidden_states = outputs[0]
            logits = self.codec_head(hidden_states)
            logits = logits.float()
            output = (logits,) + outputs[:]

            return output

        lang_model = model.talker
        num_pkv = lang_model.model.config.num_hidden_layers
        embedding_size = lang_model.model.config.hidden_size
        patch_cos_sin_cached_fp32(lang_model)
        if hasattr(lang_model, "model"):
            patch_cos_sin_cached_fp32(lang_model.model)
        lang_model._orig_forward = lang_model.forward
        lang_model.forward = types.MethodType(forward_wrap_talker, lang_model)

        pkv_shape = (
            2,
            lang_model.model.config.num_key_value_heads,
            2,
            lang_model.model.config.head_dim,
        )

        cache_position = torch.arange(2, 4)
        position_ids = cache_position.view(1, 1, -1).expand(3, 2, -1)

        input_embeds = torch.randn((2, 2, embedding_size))
        attention_mask = torch.ones([2, 4], dtype=torch.long)
        input_names = ["attention_mask", "position_ids"]
        output_names = ["logits", "hidden_states"]
        past_key_values = []
        for i in range(num_pkv):
            kv = [torch.randn(pkv_shape) for _ in range(2)]
            past_key_values.append(kv)
            input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        input_names.append("inputs_embeds")

        example_input = {
            "inputs_embeds": input_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
        }

        head_dim = lang_model.model.config.head_dim
        input_shapes = [
            ov.PartialShape([-1, -1]),
            ov.PartialShape([3, -1, -1]),
        ]
        input_shapes += (
            [
                ov.PartialShape(
                    [
                        -1,
                        lang_model.model.config.num_key_value_heads,
                        -1,
                        head_dim,
                    ]
                )
            ]
            * 2
            * num_pkv
        )
        input_shapes += [ov.PartialShape([-1, -1, input_embeds.shape[-1]])]
        __make_16bit_traceable(lang_model)

        ov_model = ov.convert_model(lang_model, example_input=example_input, input=input_shapes)
        for input, input_name in zip(ov_model.inputs, input_names):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})
        patch_stateful(ov_model, 2)
        print("✅ Talker language model successfully converted")

        if quantization_config is not None:
            print(f"⌛ Weights compression with {quantization_config['mode']} mode started")
            ov_model = nncf.compress_weights(ov_model, **quantization_config)
            print("✅ Weights compression finished")

        ov.save_model(ov_model, talker_lang_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print(f"✅ Talker model conversion finished. You can find results in {output_dir}")

    # Convert talker code predictor embedding model
    if not talker_code_predictor_embedding_path.exists():
        print("⌛ Convert talker code predictor embedding model")

        def forward_wrap_code_predictor_embedding(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            generation_steps: Optional[int] = None,
        ):
            # Get all embeddings for different generation steps
            all_embeddings = torch.stack([self.get_input_embeddings()[i](input_ids) for i in range(len(self.get_input_embeddings()))])
            # Select appropriate embedding based on generation_steps
            selected_embedding = all_embeddings[generation_steps]
            return selected_embedding

        talker_code_predictor = model.talker.code_predictor.model

        talker_code_predictor._orig_forward = talker_code_predictor.forward
        talker_code_predictor.forward = types.MethodType(forward_wrap_code_predictor_embedding, talker_code_predictor)

        __make_16bit_traceable(talker_code_predictor.get_input_embeddings())
        ov_model = ov.convert_model(
            talker_code_predictor,
            example_input={
                "input_ids": torch.ones([2, 2], dtype=torch.int64),
                "generation_steps": torch.tensor(1, dtype=torch.long),
            },
        )
        ov.save_model(ov_model, talker_code_predictor_embedding_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        talker_code_predictor.forward = talker_code_predictor._orig_forward
        print("✅ Talker Code Predictor Embedding model successfully converted")

    # Convert Talker Code Predictor model
    if not talker_code_predictor_path.exists():
        print("⌛ Convert Talker Code Predictor model")

        def forward_wrap_code_predictor(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            generation_steps: Optional[int] = None,
            **kwargs,
        ):
            if past_key_values is not None:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            # Apply projection from embedding_dim (2048) to hidden_size (1024)
            # This matches the original model's behavior in forward() method
            if inputs_embeds is not None:
                inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

            # Code predictor forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                **kwargs,
            )
            if past_key_values is not None:
                outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()

            hidden_states = outputs.last_hidden_state

            # Use torch.stack to select the appropriate head based on generation_steps
            all_logits = torch.stack([head(hidden_states) for head in self.lm_head])
            logits = all_logits[generation_steps]

            output = (logits, outputs.hidden_states[0], outputs.past_key_values)
            return output

        code_predictor_model = model.talker.code_predictor
        patch_cos_sin_cached_fp32(code_predictor_model)
        if hasattr(code_predictor_model, "model"):
            patch_cos_sin_cached_fp32(code_predictor_model.model)
        num_pkv = code_predictor_model.model.config.num_hidden_layers
        hidden_size = code_predictor_model.model.config.hidden_size
        num_code_groups = code_predictor_model.model.config.num_code_groups

        code_predictor_model._orig_forward = code_predictor_model.forward
        code_predictor_model.forward = types.MethodType(forward_wrap_code_predictor, code_predictor_model)

        head_dim = code_predictor_model.model.config.head_dim
        pkv_shape = (
            2,
            code_predictor_model.model.config.num_key_value_heads,
            2,
            head_dim,
        )

        cache_position = torch.arange(2, 4)
        position_ids = cache_position.view(1, -1)  # Code predictor uses 2D position_ids

        # Input embeds should match embedding_dim (2048) which will be projected to hidden_size (1024) inside the model
        embedding_dim = config.talker_config.hidden_size  # 2048
        input_embeds = torch.randn((2, 2, embedding_dim))
        attention_mask = torch.ones([2, 4], dtype=torch.long)
        generation_steps = torch.tensor(1, dtype=torch.long)

        input_names = ["attention_mask", "position_ids"]
        output_names = ["logits", "mid_residual_hiddens"]
        past_key_values = []
        for i in range(num_pkv):
            kv = [torch.randn(pkv_shape) for _ in range(2)]
            past_key_values.append(kv)
            input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        input_names.extend(["inputs_embeds", "generation_steps"])

        example_input = {
            "inputs_embeds": input_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "generation_steps": generation_steps,
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
                        code_predictor_model.model.config.num_key_value_heads,
                        -1,
                        head_dim,
                    ]
                )
            ]
            * 2
            * num_pkv
        )
        input_shapes += [
            ov.PartialShape([-1, -1, config.talker_config.hidden_size]),  # inputs_embeds with embedding_dim (2048)
            ov.PartialShape([]),  # generation_steps (scalar)
        ]

        __make_16bit_traceable(code_predictor_model)

        ov_model = ov.convert_model(code_predictor_model, example_input=example_input, input=input_shapes)
        for input, input_name in zip(ov_model.inputs, input_names):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})

        patch_stateful(ov_model, 2)
        print("✅ Talker Code Predictor model successfully converted")

        if quantization_config is not None:
            print(f"⌛ Weights compression with {quantization_config['mode']} mode started")
            ov_model = nncf.compress_weights(ov_model, **quantization_config)
            print("✅ Weights compression finished")

        ov.save_model(ov_model, talker_code_predictor_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print(f"✅ Talker Code Predictor model conversion finished. You can find results in {output_dir}")

    # Convert Speaker Encoder model (only for base model type)
    if config.tts_model_type == "base" and model.speaker_encoder is not None:
        if not speaker_encoder_path.exists():
            print("⌛ Convert Speaker Encoder model")
            __make_16bit_traceable(model.speaker_encoder)

            # Speaker encoder expects mel spectrogram input [batch, seq_len, mel_dim]
            mel_dim = config.speaker_encoder_config.mel_dim
            ov_model = ov.convert_model(
                model.speaker_encoder,
                example_input=torch.randn([1, 100, mel_dim], dtype=torch.float32),
                input=[ov.PartialShape([1, -1, mel_dim])],
            )
            ov.save_model(ov_model, speaker_encoder_path)
            del ov_model
            cleanup_torchscript_cache()
            gc.collect()
            print("✅ Speaker Encoder model successfully converted")

    # Convert Speech Tokenizer (if exists in model)
    # Get the local path of the model
    # Since we already loaded the model from 'ckpt' above, reuse the cached/local path
    model_id_path = Path(model_id)
    if model_id_path.exists() and model_id_path.is_dir():
        # model_id is already a local path, use it directly
        model_local_path = model_id_path
    elif use_local_dir:
        # ckpt is already a local directory path
        model_local_path = Path(ckpt)
    else:
        # ckpt is a HuggingFace model ID, model was already loaded so it's in cache
        # Use try_to_load_from_cache to get the cached path without re-downloading
        from huggingface_hub import try_to_load_from_cache

        # Try to find speech_tokenizer directory in cache
        cached_config = try_to_load_from_cache(model_id, "speech_tokenizer/config.json")
        if cached_config and cached_config != "_CACHED_NO_EXIST":
            # Get parent directory (speech_tokenizer) and its parent (model root)
            model_local_path = Path(cached_config).parent.parent
        else:
            # Fallback: download speech_tokenizer files if not in cache
            model_local_path = Path(
                snapshot_download(model_id, allow_patterns=["speech_tokenizer/**", "*.json", "*.txt"], ignore_patterns=["*.safetensors", "*.bin"])
            )

    speech_tokenizer_dir = model_local_path / "speech_tokenizer"
    speech_tokenizer_ov_dir = output_dir / "speech_tokenizer"

    if speech_tokenizer_dir.exists():
        print(f"✓ Found speech tokenizer at {speech_tokenizer_dir}")
        convert_speech_tokenizer(str(speech_tokenizer_dir), speech_tokenizer_ov_dir, use_local_dir=use_local_dir)
    else:
        print(f"ℹ️ No speech tokenizer found in model. Using PyTorch version during inference.")

    del model
    gc.collect()


def convert_speech_tokenizer(model_id, output_dir, use_local_dir=False):
    """
    Convert Qwen3-TTS speech tokenizer (encoder and decoder) to OpenVINO format.

    Args:
        model_id: HuggingFace model ID or local path to speech_tokenizer
        output_dir: Directory to save the converted models
        use_local_dir: Whether to download to local directory first
    """
    from qwen_tts.core import Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model
    from transformers import AutoConfig, AutoModel, AutoFeatureExtractor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = output_dir / SPEECH_TOKENIZER_ENCODER_NAME
    decoder_path = output_dir / SPEECH_TOKENIZER_DECODER_NAME

    if encoder_path.exists() and decoder_path.exists():
        print(f"✅ Speech tokenizer already converted. You can find results in {output_dir}")
        return

    print(f"⌛ Speech tokenizer conversion started. Be patient, it may take some time.")
    print("⌛ Load Speech tokenizer model")

    # Register config and model
    AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
    AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

    if use_local_dir:
        ckpt = Path(output_dir) / "speech_tokenizer_ckpt"
        if not ckpt.exists():
            snapshot_download(model_id, local_dir=ckpt, force_download=True)
    else:
        ckpt = model_id

    # Load model
    tokenizer_model = AutoModel.from_pretrained(ckpt, torch_dtype=torch.float32)
    tokenizer_model.eval()

    # Save feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(ckpt)
    feature_extractor.save_pretrained(output_dir)

    # Save config
    tokenizer_model.config.save_pretrained(output_dir)

    print("✅ Speech tokenizer model successfully loaded")

    # Convert encoder (MimiModel)
    if not encoder_path.exists():
        print("⌛ Convert speech tokenizer encoder")
        encoder = tokenizer_model.encoder

        # Encoder forward: takes (input_values, padding_mask) and returns audio_codes
        # input_values: [batch, 1, seq_len] audio waveform
        # We need to trace the encode path
        class EncoderWrapper(torch.nn.Module):
            def __init__(self, encoder, valid_num_quantizers):
                super().__init__()
                self.encoder = encoder
                self.valid_num_quantizers = valid_num_quantizers

            def forward(self, input_values):
                # input_values: [batch, 1, seq_len]
                encoded = self.encoder.encode(input_values=input_values, return_dict=True)
                # Return codes: [batch, num_quantizers, code_len]
                audio_codes = encoded.audio_codes[:, : self.valid_num_quantizers]
                return audio_codes

        encoder_wrapper = EncoderWrapper(encoder, tokenizer_model.encoder_valid_num_quantizers)

        # Example input: [batch=1, channels=1, seq_len=24000] (1 second at 24kHz)
        example_input = torch.randn([1, 1, 24000], dtype=torch.float32)

        __make_16bit_traceable(encoder_wrapper)
        ov_model = ov.convert_model(
            encoder_wrapper,
            example_input=example_input,
            input=[ov.PartialShape([1, 1, -1])],  # dynamic sequence length
        )

        # Set input/output names
        ov_model.inputs[0].get_tensor().set_names({"input_values"})
        ov_model.outputs[0].get_tensor().set_names({"audio_codes"})

        ov.save_model(ov_model, encoder_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Speech tokenizer encoder successfully converted")

    # Convert decoder
    if not decoder_path.exists():
        print("⌛ Convert speech tokenizer decoder")
        decoder = tokenizer_model.decoder

        # Patch masking_utils to use trace-compatible implementations.
        # transformers 4.57+ uses torch.vmap in create_causal_mask / create_sliding_window_causal_mask
        # which is incompatible with torch.jit.trace. We provide simple replacements that produce
        # identical masks without vmap.
        import transformers.masking_utils as _masking_utils

        _orig_causal = _masking_utils.create_causal_mask
        _orig_sliding = getattr(_masking_utils, "create_sliding_window_causal_mask", None)

        def _simple_causal_mask(**kwargs):
            input_embeds = kwargs["input_embeds"]
            batch_size, seq_len = input_embeds.shape[0], input_embeds.shape[1]
            dtype = input_embeds.dtype
            mask = torch.triu(
                torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=input_embeds.device),
                diagonal=1,
            )
            return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        def _simple_sliding_window_causal_mask(**kwargs):
            config = kwargs["config"]
            input_embeds = kwargs["input_embeds"]
            batch_size, seq_len = input_embeds.shape[0], input_embeds.shape[1]
            dtype = input_embeds.dtype
            window_size = getattr(config, "sliding_window", None) or 72
            mask = torch.triu(
                torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=input_embeds.device),
                diagonal=1,
            )
            sliding_mask = torch.tril(
                torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=input_embeds.device),
                diagonal=-(window_size),
            )
            mask = mask + sliding_mask
            return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        _masking_utils.create_causal_mask = _simple_causal_mask
        if _orig_sliding:
            _masking_utils.create_sliding_window_causal_mask = _simple_sliding_window_causal_mask

        try:

            class DecoderWrapper(torch.nn.Module):
                def __init__(self, decoder):
                    super().__init__()
                    self.decoder = decoder

                def forward(self, audio_codes):
                    codes_transposed = audio_codes.transpose(1, 2)
                    wav = self.decoder(codes_transposed)
                    return wav.squeeze(1)

            decoder_wrapper = DecoderWrapper(decoder)

            num_quantizers = tokenizer_model.config.decoder_config.num_quantizers
            # Trace at 325 tokens = chunk_size(300) + left_context(25), matching original chunked_decode
            example_input = torch.randint(0, 2048, [1, 325, num_quantizers], dtype=torch.long)

            traced = torch.jit.trace(decoder_wrapper, example_input)
            ov_model = ov.convert_model(
                traced,
                example_input=example_input,
                input=[ov.PartialShape([1, -1, num_quantizers])],
            )

            ov_model.inputs[0].get_tensor().set_names({"audio_codes"})
            ov_model.outputs[0].get_tensor().set_names({"audio_values"})

            ov.save_model(ov_model, decoder_path)
            del ov_model, traced
            cleanup_torchscript_cache()
            gc.collect()
            print("✅ Speech tokenizer decoder successfully converted")
        finally:
            _masking_utils.create_causal_mask = _orig_causal
            if _orig_sliding:
                _masking_utils.create_sliding_window_causal_mask = _orig_sliding

    del tokenizer_model
    gc.collect()
    print(f"✅ Speech tokenizer conversion finished. You can find results in {output_dir}")


# ============================================================================
# OVQwen3TTSModel - OpenVINO inference wrapper for Qwen3-TTS
# ============================================================================

# Define output classes for TTS models (similar to Qwen3OmniMoe)


@dataclass
class Qwen3TTSTalkerCodePredictorOutputWithPast(ModelOutput):
    """Output class for Code Predictor model."""

    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple] = None
    hidden_states: Optional[torch.FloatTensor] = None
    generation_steps: Optional[int] = None


@dataclass
class Qwen3TTSTalkerOutputWithPast(ModelOutput):
    """Output class for Talker model."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    past_hidden: Optional[torch.FloatTensor] = None
    generation_step: Optional[int] = None
    trailing_text_hidden: Optional[torch.FloatTensor] = None
    tts_pad_embed: Optional[torch.FloatTensor] = None


class OVQwen3TTSTalkerCodePredictorModelForConditionalGeneration(GenerationMixin):
    """
    OpenVINO wrapper for Qwen3-TTS Code Predictor model with GenerationMixin support.
    This model generates residual codec codes (codebook 1..N-1) based on the first codec token.
    """

    _is_stateful = False

    def __init__(self, model_dir: Path, device: str, config):
        self.model_dir = Path(model_dir)
        self.config = config

        # Load code predictor embedding model
        self.code_predictor_embedding = core.compile_model(model_dir / TALKER_CODE_PREDICTOR_EMBEDDING_NAME, device)

        # Load code predictor model
        self.model = core.read_model(model_dir / TALKER_CODE_PREDICTOR_NAME)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        compiled_model = core.compile_model(self.model, device, config={"ACTIVATIONS_SCALE_FACTOR": "8.0"} if device == "GPU" else {})
        self.request = compiled_model.create_infer_request()

        # Create embedding wrapper
        self.get_input_embeddings = lambda: self._embedding_wrapper
        self._embedding_wrapper = self._create_embedding_wrapper()

        # GenerationMixin required attributes
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self.generation_config = GenerationConfig.from_model_config(self.config) if hasattr(self.config, "to_dict") else GenerationConfig()
        self.num_pkv = 2
        self._past_length = None
        self.next_beam_idx = None
        self._skip_keys_device_placement = "past_key_values"
        self._supports_flash_attn_2 = True
        self._supports_sdpa = True
        self._supports_cache_class = True
        self._supports_static_cache = True
        self.dtype = torch.float16

    def _create_embedding_wrapper(self):
        """Create a callable wrapper for embeddings that works with OpenVINO."""

        def embedding_fn(input_ids, generation_steps):
            result = self.code_predictor_embedding(
                {
                    "input_ids": input_ids.numpy() if isinstance(input_ids, torch.Tensor) else input_ids,
                    "generation_steps": np.array(generation_steps, dtype=np.int64),
                }
            )[0]
            return torch.from_numpy(result)

        return embedding_fn

    def can_generate(self):
        """Returns True for GenerationMixin validation."""
        return True

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        generation_steps=None,
        **kwargs,
    ) -> Qwen3TTSTalkerCodePredictorOutputWithPast:
        """
        Forward pass through code predictor model.

        Args:
            generation_steps: Current generation step (0..num_code_groups-1)
        """
        # Prefill stage
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2  # hidden & layer 0
        # Generation stage
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids, generation_steps - 1)

        # Reset state if no past_key_values
        if past_key_values is None:
            self.request.reset_state()
            self.next_beam_idx = np.arange(inputs_embeds.shape[0], dtype=int)
            self._past_length = 0

        # Prepare inputs
        inputs = {
            "inputs_embeds": inputs_embeds.numpy() if isinstance(inputs_embeds, torch.Tensor) else inputs_embeds,
            "attention_mask": attention_mask.numpy() if isinstance(attention_mask, torch.Tensor) else attention_mask,
            "position_ids": position_ids.numpy() if isinstance(position_ids, torch.Tensor) else position_ids,
            "generation_steps": np.array(generation_steps, dtype=np.int64),
        }

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(inputs_embeds.shape[0], dtype=int)

        # Run inference
        self.request.start_async(inputs, share_inputs=False)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor("logits").data.copy()).to(self.device)
        hidden_states = torch.from_numpy(self.request.get_tensor("mid_residual_hiddens").data.copy()).to(self.device)

        return Qwen3TTSTalkerCodePredictorOutputWithPast(
            logits=logits,
            past_key_values=((),),
            hidden_states=hidden_states,
            generation_steps=generation_steps + 1,
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, num_new_tokens)
        model_kwargs["generation_steps"] = outputs.generation_steps
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        if past_key_values != ((),):
            past_key_values = None
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        self.next_beam_idx = np.array(beam_idx)
        return past_key_values

    def _get_past_length(self, past_key_values=None):
        return self._past_length if past_key_values else 0


class OVQwen3TTSTalkerForConditionalGeneration(GenerationMixin):
    """
    OpenVINO wrapper for Qwen3-TTS Talker model with GenerationMixin support.
    This is the main language model that generates codec tokens.
    """

    _is_stateful = False

    def __init__(self, model_dir: Path, device: str, config):
        self.model_dir = Path(model_dir)
        self.config = config
        self.device = torch.device("cpu")
        self.dtype = torch.float16

        # Load talker language model (stateful)
        self.model = core.read_model(model_dir / TALKER_LANGUAGE_NAME)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        compiled_model = core.compile_model(self.model, device)
        self.request = compiled_model.create_infer_request()

        # Load embedding models
        self.embed_tokens = core.compile_model(model_dir / TALKER_EMBEDDING_NAME, device)
        self.text_embedding = core.compile_model(model_dir / TALKER_TEXT_EMBEDDING_NAME, device)
        self.text_projection_model = core.compile_model(model_dir / TALKER_TEXT_PROJECTION_NAME, device)

        # Create embedding wrapper
        self._embedding_wrapper = self._create_embedding_wrapper()
        self._text_embedding_wrapper = self._create_text_embedding_wrapper()

        # Load code predictor
        self.code_predictor = OVQwen3TTSTalkerCodePredictorModelForConditionalGeneration(model_dir, device, config.code_predictor_config)

        # GenerationMixin required attributes
        self.main_input_name = "input_ids"
        self.generation_config = GenerationConfig.from_model_config(self.config) if hasattr(self.config, "to_dict") else GenerationConfig()
        self.num_pkv = 2
        self._past_length = None
        self.next_beam_idx = None
        self.rope_deltas = None
        self._skip_keys_device_placement = "past_key_values"
        self._supports_flash_attn_2 = True
        self._supports_sdpa = True
        self._supports_cache_class = True
        self._supports_static_cache = True

    def _create_embedding_wrapper(self):
        """Create a callable wrapper for codec embeddings."""

        def embedding_fn(input_ids):
            if isinstance(input_ids, torch.Tensor):
                # Handle scalar tensor
                if input_ids.ndim == 0:
                    input_ids = input_ids.unsqueeze(0).unsqueeze(0)
                elif input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(0)
                input_np = input_ids.numpy()
            else:
                input_np = input_ids
            result = self.embed_tokens(input_np)[0]
            return torch.from_numpy(result)

        return embedding_fn

    def _create_text_embedding_wrapper(self):
        """Create a callable wrapper for text embeddings."""

        def embedding_fn(input_ids):
            if isinstance(input_ids, torch.Tensor):
                input_np = input_ids.numpy()
            else:
                input_np = input_ids
            result = self.text_embedding(input_np)[0]
            return torch.from_numpy(result)

        return embedding_fn

    def get_input_embeddings(self):
        """Get codec input embeddings callable."""
        return self._embedding_wrapper

    def get_text_embeddings(self):
        """Get text embeddings callable."""
        return self._text_embedding_wrapper

    def text_projection(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply text projection."""
        result = self.text_projection_model(hidden_states.numpy())[0]
        return torch.from_numpy(result)

    def can_generate(self):
        """Returns True for GenerationMixin validation."""
        return True

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def get_rope_index(self, attention_mask):
        """Calculate mRoPE position IDs."""
        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        past_hidden=None,
        trailing_text_hidden=None,
        tts_pad_embed=None,
        generation_step=None,
        subtalker_dosample=None,
        subtalker_top_p=None,
        subtalker_top_k=None,
        subtalker_temperature=None,
        **kwargs,
    ) -> Qwen3TTSTalkerOutputWithPast:
        """
        Forward pass through talker model.

        Follows the same logic as the original Qwen3TTSTalkerForConditionalGeneration.forward().
        """
        # Prefill stage
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_step = -1
            codec_ids = None
        # Generation stage
        else:
            last_id_hidden = self.get_input_embeddings()(input_ids)

            # Run code predictor to generate residual codes
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=subtalker_dosample if subtalker_dosample is not None else False,
                top_p=subtalker_top_p,
                top_k=subtalker_top_k,
                temperature=subtalker_temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

            codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)

            # Aggregate codec embeddings
            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [self.code_predictor.get_input_embeddings()(predictor_result.sequences[..., i : i + 1], i) for i in range(self.config.num_code_groups - 1)],
                dim=1,
            )
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)

            # Add text hidden states
            if generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed

        # Calculate position IDs
        if attention_mask is not None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(attention_mask)
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape if input_ids is not None else (inputs_embeds.shape[0], 1)
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=self.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Reset state if no past_key_values
        if past_key_values is None:
            self.request.reset_state()
            self.next_beam_idx = np.arange(inputs_embeds.shape[0], dtype=int)
            self._past_length = 0

        # Prepare inputs
        inputs = {
            "inputs_embeds": inputs_embeds.numpy() if isinstance(inputs_embeds, torch.Tensor) else inputs_embeds,
            "attention_mask": attention_mask.numpy() if isinstance(attention_mask, torch.Tensor) else attention_mask,
            "position_ids": position_ids.numpy() if isinstance(position_ids, torch.Tensor) else position_ids,
        }

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(inputs_embeds.shape[0], dtype=int)

        # Run inference
        self.request.start_async(inputs, share_inputs=False)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor("logits").data.copy()).to(self.device)
        hidden_states = torch.from_numpy(self.request.get_tensor("hidden_states").data.copy()).to(self.device)

        return Qwen3TTSTalkerOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=((),),
            hidden_states=(hidden_states, codec_ids),
            attentions=None,
            past_hidden=hidden_states[:, -1:, :],
            generation_step=generation_step + 1,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, num_new_tokens)
        model_kwargs["hidden_states"] = outputs.hidden_states
        model_kwargs["generation_step"] = outputs.generation_step
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        hidden_states = kwargs.pop("hidden_states", None)
        if past_key_values != ((),):
            past_key_values = None
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds, cache_position, **kwargs)

        # Decode stage
        if cache_position is not None and cache_position[0] != 0:
            generation_step = kwargs.get("generation_step", 0)
            trailing_text_hidden = kwargs.get("trailing_text_hidden")
            tts_pad_embed = kwargs.get("tts_pad_embed")
            subtalker_dosample = kwargs.get("subtalker_dosample", False)
            subtalker_top_k = kwargs.get("subtalker_top_k", 50)
            subtalker_top_p = kwargs.get("subtalker_top_p", 1.0)
            subtalker_temperature = kwargs.get("subtalker_temperature", 0.9)

            inputs["past_hidden"] = hidden_states[0][:, -1:, :] if hidden_states else None
            inputs["trailing_text_hidden"] = trailing_text_hidden
            inputs["tts_pad_embed"] = tts_pad_embed
            inputs["generation_step"] = generation_step
            inputs["subtalker_dosample"] = subtalker_dosample
            inputs["subtalker_top_k"] = subtalker_top_k
            inputs["subtalker_top_p"] = subtalker_top_p
            inputs["subtalker_temperature"] = subtalker_temperature

        return inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        self.next_beam_idx = np.array(beam_idx)
        return past_key_values

    def _get_past_length(self, past_key_values=None):
        return self._past_length if past_key_values else 0


class OVQwen3TTSSpeakerEncoder:
    """
    OpenVINO wrapper for Qwen3-TTS Speaker Encoder model (for Base model type).
    """

    def __init__(self, model_dir: Path, device: str = "CPU"):
        self.model_dir = Path(model_dir)
        self.device = device

        speaker_encoder_path = self.model_dir / SPEAKER_ENCODER_NAME
        if speaker_encoder_path.exists():
            self.model = core.compile_model(speaker_encoder_path, device)
        else:
            self.model = None

    def __call__(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from mel spectrogram.

        Args:
            mel_spectrogram: Input mel spectrogram [batch, seq_len, mel_dim]

        Returns:
            Speaker embedding tensor
        """
        if self.model is None:
            raise RuntimeError("Speaker encoder model not found")

        result = self.model(mel_spectrogram.numpy())[0]
        return torch.from_numpy(result)


class OVQwen3TTSSpeechTokenizer:
    """
    OpenVINO wrapper for Qwen3-TTS Speech Tokenizer (12Hz).
    Provides encode() and decode() methods compatible with Qwen3TTSTokenizer.
    Uses OV encoder + OV decoder with chunked decoding (no PyTorch models).
    """

    # Decoder chunking parameters (matching original chunked_decode)
    DECODER_TRACE_LEN = 325  # tokens the OV decoder was traced with
    DECODER_CHUNK_SIZE = 300  # effective tokens per chunk
    DECODER_LEFT_CONTEXT = 25  # overlap tokens from previous chunk
    DECODER_UPSAMPLE = 1920  # samples per token
    DECODER_OFFSET = 555  # causal conv edge effect offset

    def __init__(self, model_dir: Path, device: str = "CPU"):
        self.model_dir = Path(model_dir)
        self.device = device

        # Load encoder (OV)
        encoder_path = self.model_dir / SPEECH_TOKENIZER_ENCODER_NAME

        if encoder_path.exists():
            self.encoder_model = core.compile_model(encoder_path, device)
        else:
            self.encoder_model = None

        # Load decoder (OV)
        self.decoder_model = None
        decoder_path = self.model_dir / SPEECH_TOKENIZER_DECODER_NAME
        if decoder_path.exists():
            self.decoder_model = core.compile_model(decoder_path, device)

        # Load config
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            import json

            with open(config_path, "r") as f:
                config_dict = json.load(f)
            self.input_sample_rate = config_dict.get("input_sample_rate", 24000)
            self.output_sample_rate = config_dict.get("output_sample_rate", 24000)
            self.encode_downsample_rate = config_dict.get("encode_downsample_rate", 1920)
            self.decode_upsample_rate = config_dict.get("decode_upsample_rate", 1920)
            self.num_quantizers = config_dict.get("decoder_config", {}).get("num_quantizers", 16)
        else:
            # Default values for 12Hz tokenizer
            self.input_sample_rate = 24000
            self.output_sample_rate = 24000
            self.encode_downsample_rate = 1920
            self.decode_upsample_rate = 1920
            self.num_quantizers = 16

        # Load feature extractor if available
        try:
            from transformers import AutoFeatureExtractor

            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
        except Exception:
            self.feature_extractor = None

    def _normalize_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize audio to target sample rate."""
        import librosa

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        if sr != self.input_sample_rate:
            audio = librosa.resample(y=audio.astype(np.float32), orig_sr=sr, target_sr=self.input_sample_rate)

        return audio.astype(np.float32)

    def encode(
        self,
        audios,
        sr: Optional[int] = None,
        return_dict: bool = True,
    ):
        """
        Encode audio waveform(s) into discrete codes.

        Args:
            audios: Audio input - can be:
                - np.ndarray: single waveform (requires sr)
                - List[np.ndarray]: list of waveforms (requires sr)
                - str: path to audio file
                - List[str]: list of audio file paths
            sr: Sample rate for numpy input
            return_dict: Whether to return a dict-like output

        Returns:
            Object with audio_codes attribute containing list of code tensors
        """
        if self.encoder_model is None:
            raise RuntimeError("Speech tokenizer encoder not loaded")

        import librosa

        # Normalize inputs to list
        if isinstance(audios, (str, np.ndarray)):
            audios = [audios]

        # Load and normalize audio
        audio_list = []
        for audio in audios:
            if isinstance(audio, str):
                # Load from file
                wav, file_sr = librosa.load(audio, sr=None, mono=True)
                wav = self._normalize_audio(wav, file_sr)
            else:
                if sr is None:
                    raise ValueError("Sample rate (sr) required for numpy input")
                wav = self._normalize_audio(audio, sr)
            audio_list.append(wav)

        # Encode each audio
        audio_codes = []
        for wav in audio_list:
            # Prepare input: [batch=1, channels=1, seq_len]
            input_values = wav.reshape(1, 1, -1)

            # Run encoder
            result = self.encoder_model({"input_values": input_values})[0]
            codes = torch.from_numpy(result[0])  # [num_quantizers, code_len]

            # Transpose to [code_len, num_quantizers] for compatibility
            codes = codes.transpose(0, 1)
            audio_codes.append(codes)

        # Return in expected format
        class EncoderOutput:
            def __init__(self, codes):
                self.audio_codes = codes

        if return_dict:
            return EncoderOutput(audio_codes)
        return (audio_codes,)

    def _chunked_ov_decode(self, codes_np):
        """
        Decode audio codes using OV decoder with chunking, matching the original
        chunked_decode(chunk_size=300, left_context_size=25) behavior.

        The OV decoder was traced at DECODER_TRACE_LEN=325 tokens. For longer sequences,
        we split into chunks of DECODER_CHUNK_SIZE=300 effective tokens with
        DECODER_LEFT_CONTEXT=25 overlap tokens from the previous chunk.

        Args:
            codes_np: [1, code_len, num_quantizers] int64 numpy array

        Returns:
            1D float32 numpy array of audio samples
        """
        code_len = codes_np.shape[1]
        wavs = []
        start = 0
        while start < code_len:
            end = min(start + self.DECODER_CHUNK_SIZE, code_len)
            ctx = self.DECODER_LEFT_CONTEXT if start > self.DECODER_LEFT_CONTEXT else start
            chunk = codes_np[:, start - ctx : end, :]
            chunk_len = chunk.shape[1]

            # Pad to DECODER_TRACE_LEN if chunk is shorter
            if chunk_len < self.DECODER_TRACE_LEN:
                pad = np.zeros((1, self.DECODER_TRACE_LEN - chunk_len, codes_np.shape[2]), dtype=np.int64)
                chunk = np.concatenate([chunk, pad], axis=1)

            ov_out = self.decoder_model({"audio_codes": chunk})[0].flatten()

            # Valid output region: discard context portion and edge-effect tail
            total_valid = chunk_len * self.DECODER_UPSAMPLE - self.DECODER_OFFSET
            context_samples = ctx * self.DECODER_UPSAMPLE
            wavs.append(ov_out[context_samples:total_valid])

            start = end

        return np.concatenate(wavs).astype(np.float32)

    def decode(
        self,
        encoded,
    ):
        """
        Decode audio codes back to waveform.

        Args:
            encoded: Can be:
                - Object with audio_codes attribute
                - Dict with "audio_codes" key
                - List of dicts with "audio_codes" key

        Returns:
            Tuple of (wavs: List[np.ndarray], sample_rate: int)
        """
        if self.decoder_model is None:
            raise RuntimeError("Speech tokenizer decoder not loaded")

        # Normalize input
        if hasattr(encoded, "audio_codes"):
            audio_codes_list = encoded.audio_codes
        elif isinstance(encoded, dict):
            audio_codes_list = [encoded["audio_codes"]]
        elif isinstance(encoded, list):
            audio_codes_list = [e["audio_codes"] for e in encoded]
        else:
            raise TypeError("encoded must be encoder output, dict, or list of dicts")

        wavs = []
        for codes in audio_codes_list:
            if isinstance(codes, torch.Tensor):
                codes_np = codes.numpy()
            elif isinstance(codes, np.ndarray):
                codes_np = codes
            else:
                codes_np = np.array(codes)

            # Ensure shape is [1, code_len, num_quantizers]
            if codes_np.ndim == 2:
                codes_np = codes_np[np.newaxis, ...]
            elif codes_np.ndim == 3 and codes_np.shape[0] != 1:
                codes_np = codes_np[:1]

            wav = self._chunked_ov_decode(codes_np.astype(np.int64))
            wavs.append(wav)

        return wavs, self.output_sample_rate

    def get_model_type(self) -> str:
        return "qwen3_tts_tokenizer_12hz"

    def get_input_sample_rate(self) -> int:
        return self.input_sample_rate

    def get_output_sample_rate(self) -> int:
        return self.output_sample_rate

    def get_encode_downsample_rate(self) -> int:
        return self.encode_downsample_rate

    def get_decode_upsample_rate(self) -> int:
        return self.decode_upsample_rate


class OVQwen3TTSModel:
    """
    OpenVINO inference wrapper for Qwen3-TTS models.

    Provides the same interface as Qwen3TTSModel for:
      - CustomVoice: generate_custom_voice()
      - VoiceDesign: generate_voice_design()
      - Base: generate_voice_clone() + create_voice_clone_prompt()

    Usage:
        model = OVQwen3TTSModel.from_pretrained("/path/to/converted/model", device="CPU")
        wavs, sr = model.generate_custom_voice(
            text="Hello, world!",
            language="English",
            speaker="Vivian",
        )
    """

    def __init__(
        self,
        model_dir: Path,
        processor,
        speech_tokenizer,
        generate_defaults: Optional[dict] = None,
        device: str = "CPU",
    ):
        # Import here to avoid unnecessary dependencies at module load time
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

        self.model_dir = Path(model_dir)
        self.device = device
        self.processor = processor
        self.speech_tokenizer = speech_tokenizer
        self.generate_defaults = generate_defaults or {}

        # Load config
        self.config = Qwen3TTSConfig.from_pretrained(model_dir)

        # Initialize talker using GenerationMixin wrapper
        self.talker = OVQwen3TTSTalkerForConditionalGeneration(model_dir, device, self.config.talker_config)

        # Initialize speaker encoder (for base model)
        if self.config.tts_model_type == "base":
            self.speaker_encoder = OVQwen3TTSSpeakerEncoder(model_dir, device)
        else:
            self.speaker_encoder = None

        # Model properties
        self.tokenizer_type = self.config.tokenizer_type
        self.tts_model_size = self.config.tts_model_size
        self.tts_model_type = self.config.tts_model_type
        self.speaker_encoder_sample_rate = self.config.speaker_encoder_config.sample_rate

        # Supported speakers and languages
        self.supported_speakers = set(self.config.talker_config.spk_id.keys())
        self.supported_languages = {"auto"}
        for language_id in self.config.talker_config.codec_language_id.keys():
            if "dialect" not in language_id:
                self.supported_languages.add(language_id)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "CPU",
        checkpoint_path: str = None,
        **kwargs,
    ) -> "OVQwen3TTSModel":
        """
        Load a converted Qwen3-TTS OpenVINO model.

        Args:
            model_dir: Path to the converted OpenVINO model directory
            device: OpenVINO device to use (CPU, GPU, etc.)
            checkpoint_path: Path to the original PyTorch checkpoint (for loading processor/tokenizer).
                           If None, will try to load from model_dir or a saved checkpoint_path.txt
            **kwargs: Additional arguments

        Returns:
            OVQwen3TTSModel instance
        """
        # Import here to avoid unnecessary dependencies at module load time
        from qwen_tts.core.models import Qwen3TTSProcessor
        import json

        model_dir = Path(model_dir)

        # Determine checkpoint path for loading processor
        if checkpoint_path is None:
            # Try to load from saved checkpoint_path.txt
            checkpoint_path_file = model_dir / "checkpoint_path.txt"
            if checkpoint_path_file.exists():
                with open(checkpoint_path_file, "r") as f:
                    checkpoint_path = f.read().strip()
                print(f"Loading processor from saved checkpoint: {checkpoint_path}")
            else:
                # Fall back to model_dir
                checkpoint_path = str(model_dir)
                print(f"No checkpoint_path.txt found, trying to load processor from model_dir")

        # Load processor
        processor = Qwen3TTSProcessor.from_pretrained(checkpoint_path, fix_mistral_regex=True)

        # Load speech tokenizer - prefer OpenVINO version if available
        speech_tokenizer = None
        speech_tokenizer_ov_path = model_dir / "speech_tokenizer"
        encoder_path = speech_tokenizer_ov_path / SPEECH_TOKENIZER_ENCODER_NAME
        decoder_path = speech_tokenizer_ov_path / SPEECH_TOKENIZER_DECODER_NAME

        if encoder_path.exists() and decoder_path.exists():
            # Use OpenVINO speech tokenizer
            print("Loading OpenVINO speech tokenizer")
            speech_tokenizer = OVQwen3TTSSpeechTokenizer(speech_tokenizer_ov_path, device)
        else:
            # Try PyTorch speech tokenizer
            try:
                from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

                if speech_tokenizer_ov_path.exists():
                    speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(speech_tokenizer_ov_path)
                    print("Loaded PyTorch speech tokenizer from model_dir/speech_tokenizer")
                else:
                    # Try to find speech tokenizer in parent or sibling directories
                    for potential_path in [model_dir.parent / "speech_tokenizer", model_dir / "ckpt" / "speech_tokenizer"]:
                        if potential_path.exists():
                            speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(potential_path)
                            print(f"Loaded PyTorch speech tokenizer from {potential_path}")
                            break
            except ImportError:
                pass

            if speech_tokenizer is None:
                print("Warning: Speech tokenizer not found. Voice synthesis will not work.")

        # Load generation config
        generate_config_path = model_dir / "generation_config.json"
        generate_defaults = {}
        if generate_config_path.exists():
            with open(generate_config_path, "r", encoding="utf-8") as f:
                generate_defaults = json.load(f)

        return cls(
            model_dir=model_dir,
            processor=processor,
            speech_tokenizer=speech_tokenizer,
            generate_defaults=generate_defaults,
            device=device,
        )

    def get_supported_speakers(self) -> list:
        """Get list of supported speaker names."""
        return sorted(self.supported_speakers)

    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return sorted(self.supported_languages)

    def _validate_languages(self, languages: list) -> None:
        """Validate that requested languages are supported."""
        for lang in languages:
            if lang is not None and lang.lower() not in self.supported_languages:
                raise ValueError(f"Unsupported language: {lang}. Supported: {sorted(self.supported_languages)}")

    def _validate_speakers(self, speakers: list) -> None:
        """Validate that requested speakers are supported."""
        for spk in speakers:
            if spk is not None and spk != "" and spk.lower() not in self.supported_speakers:
                raise ValueError(f"Unsupported speaker: {spk}. Supported: {sorted(self.supported_speakers)}")

    def _ensure_list(self, x) -> list:
        """Ensure input is a list."""
        return x if isinstance(x, list) else [x]

    def _build_assistant_text(self, text: str) -> str:
        """Build assistant text format."""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        """Build reference text format for voice clone."""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _build_instruct_text(self, instruct: str) -> str:
        """Build instruction text format."""
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize_texts(self, texts: list) -> list:
        """Tokenize list of texts."""
        input_ids = []
        for text in texts:
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            input_id = inputs["input_ids"]
            input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
            input_ids.append(input_id)
        return input_ids

    def _merge_generate_kwargs(
        self,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """Merge user-provided generation arguments with defaults."""
        hard_defaults = dict(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            max_new_tokens=4096,
        )

        def pick(name: str, user_val):
            if user_val is not None:
                return user_val
            if name in self.generate_defaults:
                return self.generate_defaults[name]
            return hard_defaults[name]

        merged = dict(kwargs)
        merged.update(
            do_sample=pick("do_sample", do_sample),
            top_k=pick("top_k", top_k),
            top_p=pick("top_p", top_p),
            temperature=pick("temperature", temperature),
            repetition_penalty=pick("repetition_penalty", repetition_penalty),
            subtalker_dosample=pick("subtalker_dosample", subtalker_dosample),
            subtalker_top_k=pick("subtalker_top_k", subtalker_top_k),
            subtalker_top_p=pick("subtalker_top_p", subtalker_top_p),
            subtalker_temperature=pick("subtalker_temperature", subtalker_temperature),
            max_new_tokens=pick("max_new_tokens", max_new_tokens),
        )
        return merged

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        generated_tokens: Optional[list] = None,
    ) -> torch.Tensor:
        """Sample next token from logits."""
        # Apply repetition penalty
        if generated_tokens is not None and repetition_penalty != 1.0:
            for token in set(generated_tokens):
                logits[:, :, token] /= repetition_penalty

        if not do_sample:
            return logits.argmax(dim=-1)

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
        return next_token.view(logits.shape[:-1])

    @torch.inference_mode()
    def generate_custom_voice(
        self,
        text,
        speaker,
        language=None,
        instruct=None,
        non_streaming_mode: bool = True,
        **kwargs,
    ):
        """
        Generate speech with the CustomVoice model using a predefined speaker id.

        Args:
            text: Text(s) to synthesize
            speaker: Speaker name(s)
            language: Language(s) for each sample
            instruct: Optional instruction(s)
            non_streaming_mode: Whether to use non-streaming mode
            **kwargs: Generation parameters

        Returns:
            Tuple of (wavs, sample_rate)
        """
        if self.tts_model_type != "custom_voice":
            raise ValueError(f"Model type {self.tts_model_type} does not support generate_custom_voice. " "Please use a CustomVoice model.")

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        speakers = self._ensure_list(speaker)
        instructs = self._ensure_list(instruct) if isinstance(instruct, list) else ([instruct] * len(texts) if instruct is not None else [""] * len(texts))

        # Expand single values to match batch size
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(speakers) == 1 and len(texts) > 1:
            speakers = speakers * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        # Validate batch sizes
        if not (len(texts) == len(languages) == len(speakers) == len(instructs)):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}, " f"speaker={len(speakers)}, instruct={len(instructs)}")

        self._validate_languages(languages)
        self._validate_speakers(speakers)

        # Tokenize texts
        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        # Tokenize instructions
        instruct_ids = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        # Generate talker codes
        talker_codes_list = self._generate_talker_codes(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        # Decode codes to waveforms
        if self.speech_tokenizer is None:
            raise RuntimeError("Speech tokenizer not loaded. Cannot decode audio.")

        wavs, fs = self.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs

    @torch.inference_mode()
    def generate_voice_design(
        self,
        text,
        language=None,
        instruct=None,
        non_streaming_mode: bool = True,
        **kwargs,
    ):
        """
        Generate speech with the VoiceDesign model using natural language instructions.

        Args:
            text: Text(s) to synthesize
            language: Language(s) for each sample
            instruct: Voice design instruction(s) describing desired voice characteristics
            non_streaming_mode: Whether to use non-streaming mode
            **kwargs: Generation parameters

        Returns:
            Tuple of (wavs, sample_rate)
        """
        if self.tts_model_type != "voice_design":
            raise ValueError(f"Model type {self.tts_model_type} does not support generate_voice_design. " "Please use a VoiceDesign model.")

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        instructs = self._ensure_list(instruct) if isinstance(instruct, list) else ([instruct] * len(texts) if instruct is not None else [""] * len(texts))

        # Expand single values to match batch size
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        # Validate batch sizes
        if not (len(texts) == len(languages) == len(instructs)):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}, " f"instruct={len(instructs)}")

        self._validate_languages(languages)

        # Tokenize texts
        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        # Tokenize instructions
        instruct_ids = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        # Generate talker codes (VoiceDesign doesn't use speaker parameter)
        talker_codes_list = self._generate_talker_codes(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=None,  # VoiceDesign doesn't use predefined speakers
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        # Decode codes to waveforms
        if self.speech_tokenizer is None:
            raise RuntimeError("Speech tokenizer not loaded. Cannot decode audio.")

        wavs, fs = self.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs

    def _load_audio_to_np(self, audio_path: str):
        """Load audio from file path or URL to numpy array."""
        import librosa
        import io
        import base64
        import urllib.request
        from urllib.parse import urlparse

        def is_url(s):
            try:
                u = urlparse(s)
                return u.scheme in ("http", "https") and bool(u.netloc)
            except:
                return False

        def is_base64(s):
            if s.startswith("data:audio"):
                return True
            if ("/" not in s and "\\" not in s) and len(s) > 256:
                return True
            return False

        if is_url(audio_path):
            import soundfile as sf

            with urllib.request.urlopen(audio_path) as resp:  # nosec B310 - audio URL from internal pipeline input
                audio_bytes = resp.read()
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        elif is_base64(audio_path):
            import soundfile as sf

            if "," in audio_path and audio_path.strip().startswith("data:"):
                audio_path = audio_path.split(",", 1)[1]
            wav_bytes = base64.b64decode(audio_path)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios):
        """Normalize audio inputs to list of (waveform, sr) tuples."""
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")

        for i, (audio, sr) in enumerate(out):
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1).astype(np.float32)
                out[i] = (audio, sr)

        return out

    def extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Extract speaker embedding from audio using the speaker encoder.

        Args:
            audio: Audio waveform as numpy array (should be at 24kHz)
            sr: Sample rate of the audio (should be 24000)

        Returns:
            Speaker embedding tensor
        """
        if self.speaker_encoder is None or self.speaker_encoder.model is None:
            raise RuntimeError("Speaker encoder not available for this model type")

        import librosa
        from librosa.filters import mel as librosa_mel_fn

        # Resample if necessary
        if sr != self.speaker_encoder_sample_rate:
            audio = librosa.resample(y=audio.astype(np.float32), orig_sr=sr, target_sr=self.speaker_encoder_sample_rate)
            sr = self.speaker_encoder_sample_rate

        # Compute mel spectrogram using the same parameters as the original model
        # Parameters from modeling_qwen3_tts.py extract_speaker_embedding:
        # n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000
        n_fft = 1024
        num_mels = 128
        hop_size = 256
        win_size = 1024
        fmin = 0
        fmax = 12000

        # Convert to tensor
        y = torch.from_numpy(audio).unsqueeze(0).float()

        # Get mel filterbank
        mel_basis = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis = torch.from_numpy(mel_basis).float()

        # Compute STFT
        hann_window = torch.hann_window(win_size)
        padding = (n_fft - hop_size) // 2
        y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.abs(spec)

        # Apply mel filterbank
        spec = torch.matmul(mel_basis, spec)

        # Dynamic range compression (log)
        spec = torch.log(torch.clamp(spec, min=1e-5))

        # Transpose to [batch, time, mel_dim]
        mels = spec.transpose(1, 2)

        # Extract embedding using OpenVINO model
        embedding = self.speaker_encoder(mels)

        return embedding.squeeze(0)

    def create_voice_clone_prompt(
        self,
        ref_audio,
        ref_text=None,
        x_vector_only_mode=False,
    ):
        """
        Build voice-clone prompt items from reference audio (and optionally reference text) using Base model.

        Modes:
          - x_vector_only_mode=True:
              Only speaker embedding is used to clone voice; ref_text/ref_code are ignored.
          - x_vector_only_mode=False:
              ICL mode is enabled automatically. In this case ref_text is required.

        Args:
            ref_audio: Reference audio - can be:
                - str: wav path / URL / base64
                - (np.ndarray, sr): waveform + sampling rate
                - list of the above
            ref_text: Reference transcript(s). Required when x_vector_only_mode=False.
            x_vector_only_mode: Whether to use speaker embedding only.

        Returns:
            List of VoiceClonePromptItem dicts
        """
        import librosa

        if self.tts_model_type != "base":
            raise ValueError(f"Model type {self.tts_model_type} does not support create_voice_clone_prompt. " "Please use a Base model.")

        ref_audio_list = self._ensure_list(ref_audio)
        ref_text_list = self._ensure_list(ref_text) if isinstance(ref_text, list) else ([ref_text] * len(ref_audio_list))
        xvec_list = self._ensure_list(x_vector_only_mode) if isinstance(x_vector_only_mode, list) else ([x_vector_only_mode] * len(ref_audio_list))

        if len(ref_text_list) != len(ref_audio_list) or len(xvec_list) != len(ref_audio_list):
            raise ValueError(f"Batch size mismatch: ref_audio={len(ref_audio_list)}, ref_text={len(ref_text_list)}, x_vector_only_mode={len(xvec_list)}")

        # Normalize audio inputs
        normalized = self._normalize_audio_inputs(ref_audio_list)

        # Encode reference audio to codes
        ref_wavs = [wav for wav, sr in normalized]
        ref_srs = [sr for wav, sr in normalized]

        if len(set(ref_srs)) == 1:
            enc = self.speech_tokenizer.encode(ref_wavs, sr=ref_srs[0])
            ref_codes = enc.audio_codes
        else:
            ref_codes = []
            for wav, sr in normalized:
                enc = self.speech_tokenizer.encode(wav, sr=sr)
                ref_codes.append(enc.audio_codes[0])

        # Build prompt items
        items = []
        for i, ((wav, sr), code, rtext, xvec_only) in enumerate(zip(normalized, ref_codes, ref_text_list, xvec_list)):
            if not xvec_only:
                if rtext is None or rtext == "":
                    raise ValueError(f"ref_text is required when x_vector_only_mode=False (ICL mode). Bad index={i}")

            # Resample for speaker encoder
            wav_resample = wav
            if sr != self.speaker_encoder_sample_rate:
                wav_resample = librosa.resample(y=wav.astype(np.float32), orig_sr=int(sr), target_sr=self.speaker_encoder_sample_rate)

            # Extract speaker embedding
            spk_emb = self.extract_speaker_embedding(wav_resample, self.speaker_encoder_sample_rate)

            items.append(
                {
                    "ref_code": None if xvec_only else code,
                    "ref_spk_embedding": spk_emb,
                    "x_vector_only_mode": bool(xvec_only),
                    "icl_mode": bool(not xvec_only),
                    "ref_text": rtext,
                }
            )

        return items

    def _prompt_items_to_voice_clone_prompt(self, items):
        """Convert list of prompt items to voice_clone_prompt dict."""
        return {
            "ref_code": [it["ref_code"] for it in items],
            "ref_spk_embedding": [it["ref_spk_embedding"] for it in items],
            "x_vector_only_mode": [it["x_vector_only_mode"] for it in items],
            "icl_mode": [it["icl_mode"] for it in items],
        }

    def generate_voice_clone(
        self,
        text,
        language=None,
        ref_audio=None,
        ref_text=None,
        x_vector_only_mode=False,
        voice_clone_prompt=None,
        non_streaming_mode: bool = False,
        **kwargs,
    ):
        """
        Voice clone speech using the Base model.

        You can provide either:
          - (ref_audio, ref_text, x_vector_only_mode) and let this method build the prompt, OR
          - voice_clone_prompt returned by create_voice_clone_prompt

        Args:
            text: Text(s) to synthesize.
            language: Language(s) for each sample.
            ref_audio: Reference audio(s) for prompt building.
            ref_text: Reference text(s) used for ICL mode.
            x_vector_only_mode: If True, only speaker embedding is used.
            voice_clone_prompt: Pre-built prompt from create_voice_clone_prompt.
            non_streaming_mode: Using non-streaming text input.
            **kwargs: Generation parameters.

        Returns:
            Tuple of (wavs: List[np.ndarray], sample_rate: int)
        """
        if self.tts_model_type != "base":
            raise ValueError(f"Model type {self.tts_model_type} does not support generate_voice_clone. " "Please use a Base model.")

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(texts) != len(languages):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}")

        self._validate_languages(languages)

        # Build or use provided prompt
        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either voice_clone_prompt or ref_audio must be provided.")
            prompt_items = self.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode)
            if len(prompt_items) == 1 and len(texts) > 1:
                prompt_items = prompt_items * len(texts)
            if len(prompt_items) != len(texts):
                raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [it["ref_text"] for it in prompt_items]
        else:
            if isinstance(voice_clone_prompt, list):
                prompt_items = voice_clone_prompt
                if len(prompt_items) == 1 and len(texts) > 1:
                    prompt_items = prompt_items * len(texts)
                if len(prompt_items) != len(texts):
                    raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
                ref_texts_for_ids = [it["ref_text"] for it in prompt_items]
            else:
                voice_clone_prompt_dict = voice_clone_prompt
                ref_texts_for_ids = None

        # Tokenize texts
        input_texts = [self._build_assistant_text(t) for t in texts]
        input_ids = self._tokenize_texts(input_texts)

        # Tokenize reference texts
        ref_ids = None
        if ref_texts_for_ids is not None:
            ref_ids = []
            for rt in ref_texts_for_ids:
                if rt is None or rt == "":
                    ref_ids.append(None)
                else:
                    ref_tok = self._tokenize_texts([self._build_ref_text(rt)])[0]
                    ref_ids.append(ref_tok)

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        # Generate talker codes
        talker_codes_list = self._generate_talker_codes(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        # Concatenate ref codes with generated codes for decoding
        codes_for_decode = []
        for i, codes in enumerate(talker_codes_list):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                ref_code = ref_code_list[i]
                if isinstance(ref_code, torch.Tensor):
                    codes_for_decode.append(torch.cat([ref_code.to(codes.device), codes], dim=0))
                else:
                    codes_for_decode.append(torch.cat([torch.from_numpy(ref_code), codes], dim=0))
            else:
                codes_for_decode.append(codes)

        # Decode codes to waveforms
        if self.speech_tokenizer is None:
            raise RuntimeError("Speech tokenizer not loaded. Cannot decode audio.")

        wavs_all, fs = self.speech_tokenizer.decode([{"audio_codes": c} for c in codes_for_decode])

        # Trim ref audio from output
        wavs_out = []
        for i, wav in enumerate(wavs_all):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                ref_len = int(ref_code_list[i].shape[0] if hasattr(ref_code_list[i], "shape") else len(ref_code_list[i]))
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wavs_out.append(wav[cut:])
            else:
                wavs_out.append(wav)

        return wavs_out, fs

    def _generate_talker_codes(
        self,
        input_ids: list,
        instruct_ids: Optional[list] = None,
        ref_ids: Optional[list] = None,
        voice_clone_prompt: Optional[dict] = None,
        languages: Optional[list] = None,
        speakers: Optional[list] = None,
        non_streaming_mode: bool = True,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        **kwargs,
    ) -> list:
        """
        Generate talker codes using OpenVINO models with GenerationMixin.

        This method follows the original Qwen3TTSForConditionalGeneration.generate() flow,
        using self.talker.generate() instead of manual loop.
        """
        batch_size = len(input_ids)
        talker_codes_list = []

        # Process each sample in batch
        for idx in range(batch_size):
            # Reset KV cache states
            self.talker.request.reset_state()
            self.talker.code_predictor.request.reset_state()
            self.talker.rope_deltas = None

            input_id = input_ids[idx]
            language = languages[idx] if languages else "Auto"
            speaker = speakers[idx] if speakers else None
            instruct_id = instruct_ids[idx] if instruct_ids else None
            ref_id = ref_ids[idx] if ref_ids else None

            # Check if this is a voice clone request
            is_voice_clone = voice_clone_prompt is not None
            ref_spk_embedding = None
            x_vector_only_mode = False
            icl_mode = False

            if is_voice_clone:
                ref_spk_embedding_list = voice_clone_prompt.get("ref_spk_embedding", [])
                x_vector_only_mode_list = voice_clone_prompt.get("x_vector_only_mode", [])
                icl_mode_list = voice_clone_prompt.get("icl_mode", [])

                if idx < len(ref_spk_embedding_list):
                    ref_spk_embedding = ref_spk_embedding_list[idx]
                if idx < len(x_vector_only_mode_list):
                    x_vector_only_mode = x_vector_only_mode_list[idx]
                if idx < len(icl_mode_list):
                    icl_mode = icl_mode_list[idx]

            # Get special embeddings
            special_token_ids = torch.tensor(
                [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                dtype=input_id.dtype,
            )
            special_embeds = self.talker.text_projection(self.talker.get_text_embeddings()(special_token_ids))
            tts_bos_embed, tts_eos_embed, tts_pad_embed = special_embeds.chunk(3, dim=1)

            # Get language ID
            language_id = None
            if language.lower() != "auto":
                if language.lower() in self.config.talker_config.codec_language_id:
                    language_id = self.config.talker_config.codec_language_id[language.lower()]

            # Handle dialect
            if language.lower() in ["chinese", "auto"] and speaker and speaker.lower() in self.config.talker_config.spk_is_dialect:
                dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
                if dialect:
                    language_id = self.config.talker_config.codec_language_id[dialect]

            # Get speaker embedding
            speaker_embed = None
            if ref_spk_embedding is not None:
                # Voice clone mode: use provided speaker embedding
                if isinstance(ref_spk_embedding, torch.Tensor):
                    speaker_embed = ref_spk_embedding.view(1, 1, -1)
                else:
                    speaker_embed = torch.from_numpy(ref_spk_embedding).view(1, 1, -1)
            elif speaker and speaker != "" and speaker.lower() in self.config.talker_config.spk_id:
                spk_id = self.config.talker_config.spk_id[speaker.lower()]
                speaker_embed = self.talker.get_input_embeddings()(torch.tensor([[spk_id]], dtype=input_id.dtype))

            # Build codec prefill
            if language_id is None:
                codec_prefill_list = [
                    [
                        self.config.talker_config.codec_nothink_id,
                        self.config.talker_config.codec_think_bos_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]
                ]
            else:
                codec_prefill_list = [
                    [
                        self.config.talker_config.codec_think_id,
                        self.config.talker_config.codec_think_bos_id,
                        language_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]
                ]

            codec_input_embedding_0 = self.talker.get_input_embeddings()(torch.tensor(codec_prefill_list, dtype=input_id.dtype))
            codec_input_embedding_1 = self.talker.get_input_embeddings()(
                torch.tensor(
                    [
                        [
                            self.config.talker_config.codec_pad_id,
                            self.config.talker_config.codec_bos_id,
                        ]
                    ],
                    dtype=input_id.dtype,
                )
            )

            if speaker_embed is None:
                codec_input_embedding = torch.cat([codec_input_embedding_0, codec_input_embedding_1], dim=1)
            else:
                codec_input_embedding = torch.cat([codec_input_embedding_0, speaker_embed.view(1, 1, -1), codec_input_embedding_1], dim=1)

            # Build talker input embeddings
            # Role embedding: <|im_start|>assistant\n
            talker_input_embed_role = self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, :3]))

            # TTS embeddings
            talker_input_embed = (
                torch.cat(
                    [
                        tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
                        tts_bos_embed,
                    ],
                    dim=1,
                )
                + codec_input_embedding[:, :-1]
            )

            talker_input_embed = torch.cat([talker_input_embed_role, talker_input_embed], dim=1)

            # Add instruct embeddings if provided
            if instruct_id is not None:
                instruct_embed = self.talker.text_projection(self.talker.get_text_embeddings()(instruct_id))
                talker_input_embed = torch.cat([instruct_embed, talker_input_embed], dim=1)

            # ICL mode (voice clone with ref audio + ref text) or normal mode
            # These branches are MUTUALLY EXCLUSIVE — ICL mode handles both ref and target text
            if icl_mode and ref_id is not None:
                ref_code_list = voice_clone_prompt.get("ref_code", [])
                has_ref_code = idx < len(ref_code_list) and ref_code_list[idx] is not None

                if has_ref_code:
                    # Replicate original generate_icl_prompt():
                    # 1. text_embed = ref_text + target_text + eos (concatenated)
                    text_id = input_id[:, 3:-5]  # strip <|im_start|>assistant\n ... <|im_end|>\n
                    ref_id_stripped = ref_id[:, 3:-2]  # strip <|im_start|>assistant\n ... \n
                    text_embed = self.talker.text_projection(self.talker.get_text_embeddings()(torch.cat([ref_id_stripped, text_id], dim=-1)))
                    text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)

                    # 2. codec_embed = sum of all code group embeddings + codec_bos prefix
                    ref_code = ref_code_list[idx]
                    if not isinstance(ref_code, torch.Tensor):
                        ref_code = torch.from_numpy(ref_code)
                    # ref_code shape: [code_len, num_quantizers]
                    num_code_groups = self.config.talker_config.num_code_groups
                    codec_embeds = []
                    for i in range(num_code_groups):
                        code_slice = ref_code[:, i : i + 1].long()  # [code_len, 1]
                        if i == 0:
                            codec_embeds.append(self.talker.get_input_embeddings()(code_slice))
                        else:
                            # OV code_predictor embedding: callable(input_ids, generation_steps)
                            codec_embeds.append(self.talker.code_predictor.get_input_embeddings()(code_slice, i - 1))
                    # Sum across code groups: each is [code_len, 1, D] -> cat on dim=1 -> [code_len, num_groups, D] -> sum -> [code_len, D]
                    codec_embed = torch.cat(codec_embeds, dim=1).sum(1).unsqueeze(0)  # [1, code_len, D]
                    codec_bos_embed = self.talker.get_input_embeddings()(torch.tensor([[self.config.talker_config.codec_bos_id]], dtype=input_id.dtype))
                    codec_embed = torch.cat([codec_bos_embed, codec_embed], dim=1)  # [1, 1+code_len, D]

                    # 3. Build ICL input embed based on streaming mode
                    text_lens = text_embed.shape[1]
                    codec_lens = codec_embed.shape[1]

                    if non_streaming_mode:
                        # text side: text_embed + codec_pad for each position
                        icl_text_part = text_embed + self.talker.get_input_embeddings()(
                            torch.tensor([[self.config.talker_config.codec_pad_id] * text_lens], dtype=input_id.dtype)
                        )
                        # codec side: codec_embed + tts_pad for each position
                        icl_codec_part = codec_embed + tts_pad_embed.expand(-1, codec_lens, -1)
                        icl_input_embed = torch.cat([icl_text_part, icl_codec_part], dim=1)
                        trailing_text_hidden = tts_pad_embed
                    else:
                        # Streaming mode: blend based on length
                        if text_lens > codec_lens:
                            icl_input_embed = text_embed[:, :codec_lens] + codec_embed
                            trailing_text_hidden = text_embed[:, codec_lens:]
                        else:
                            text_embed_padded = torch.cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
                            icl_input_embed = text_embed_padded + codec_embed
                            trailing_text_hidden = tts_pad_embed

                    talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
                else:
                    # ICL mode but no ref_code — fall through to normal handling
                    icl_mode = False

            if not icl_mode or ref_id is None:
                # Normal (non-ICL) text handling
                # Add first text token + last codec embedding
                talker_input_embed = torch.cat(
                    [talker_input_embed, self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input_embedding[:, -1:]],
                    dim=1,
                )

                if non_streaming_mode:
                    talker_input_embed = talker_input_embed[:, :-1]  # Remove the just-added text token
                    text_embed = self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:-5]))
                    text_embed_with_eos = torch.cat([text_embed, tts_eos_embed], dim=1)
                    codec_pad_embed = self.talker.get_input_embeddings()(
                        torch.tensor([[self.config.talker_config.codec_pad_id] * text_embed_with_eos.shape[1]], dtype=input_id.dtype)
                    )
                    codec_bos_embed = self.talker.get_input_embeddings()(torch.tensor([[self.config.talker_config.codec_bos_id]], dtype=input_id.dtype))
                    talker_input_embed = torch.cat(
                        [
                            talker_input_embed,
                            text_embed_with_eos + codec_pad_embed,
                            tts_pad_embed + codec_bos_embed,
                        ],
                        dim=1,
                    )
                    trailing_text_hidden = tts_pad_embed
                else:
                    # Streaming mode
                    trailing_text_hidden = torch.cat(
                        [self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 4:-5])), tts_eos_embed],
                        dim=1,
                    )

            # Build attention mask
            seq_len = talker_input_embed.shape[1]
            attention_mask = torch.ones([1, seq_len], dtype=torch.long)

            # Build suppress_tokens list: suppress top 1024 vocab IDs except EOS
            # (matches original Qwen3TTSForConditionalGeneration.generate behavior)
            suppress_tokens = [
                i
                for i in range(self.config.talker_config.vocab_size - 1024, self.config.talker_config.vocab_size)
                if i not in (self.config.talker_config.codec_eos_token_id,)
            ]

            # Setup generation kwargs for talker
            talker_kwargs = dict(
                inputs_embeds=talker_input_embed,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.config.talker_config.codec_eos_token_id,
                pad_token_id=self.config.talker_config.codec_pad_id,
                suppress_tokens=suppress_tokens,
                # Custom kwargs for forward
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                subtalker_dosample=subtalker_dosample,
                subtalker_top_k=subtalker_top_k,
                subtalker_top_p=subtalker_top_p,
                subtalker_temperature=subtalker_temperature,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

            # Generate using GenerationMixin
            generation_output = self.talker.generate(**talker_kwargs)

            # Extract talker codes from output
            # The hidden_states tuple contains (hidden_states, codec_ids) for each step
            # codec_ids has shape [batch, num_code_groups]
            generated_codes = []
            if hasattr(generation_output, "hidden_states") and generation_output.hidden_states:
                for step_hidden in generation_output.hidden_states:
                    if isinstance(step_hidden, tuple) and len(step_hidden) == 2:
                        codec_ids = step_hidden[1]
                        if codec_ids is not None:
                            generated_codes.append(codec_ids.squeeze(0))

            if len(generated_codes) > 0:
                talker_codes = torch.stack(generated_codes, dim=0)
            else:
                # Fallback: use sequences if no codec_ids in hidden states
                talker_codes = generation_output.sequences[:, 1:].T  # Remove first token, transpose

            talker_codes_list.append(talker_codes)

        return talker_codes_list
