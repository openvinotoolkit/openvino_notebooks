import gc
import sys
import types
from pathlib import Path
from typing import Any, Optional, Callable, Tuple

import numpy as np
import openvino as ov

# Import torch and transformers only when needed for conversion
try:
    import torch
    from huggingface_hub import snapshot_download
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from torch import nn
    from transformers.cache_utils import DynamicCache, DynamicLayer
    from torch._dynamo import is_compiling as is_torchdynamo_compiling
    from transformers.utils import is_torch_xpu_available

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Define dummy functions for when torch is not available
    def is_torchdynamo_compiling():
        return False

    is_torch_xpu_available = False

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

# Add path for Qwen3-ASR module
sys.path.insert(0, str(Path(__file__).parent / "Qwen3-ASR"))

# Import Qwen3-ASR only when torch is available (for conversion)
if TORCH_AVAILABLE:
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRThinkerForConditionalGeneration,
        Qwen3ASRAudioEncoder,
    )
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRConfig,
        Qwen3ASRThinkerConfig,
        Qwen3ASRAudioEncoderConfig,
    )
    from transformers import AutoProcessor


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
    if not TORCH_AVAILABLE:
        return
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


ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask_without_vmap)

ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", sdpa_mask_without_vmap)


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

# File naming conventions for Qwen3-ASR
THINKER_LANGUAGE_NAME = "openvino_thinker_language_model.xml"
THINKER_AUDIO_NAME = "openvino_thinker_audio_model.xml"
THINKER_AUDIO_ENCODER_NAME = "openvino_thinker_audio_encoder_model.xml"
THINKER_EMBEDDING_NAME = "openvino_thinker_embedding_model.xml"


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


def convert_qwen3_asr_model(model_id, output_dir, quantization_config=None, use_local_dir=False):
    """
    Convert Qwen3-ASR model to OpenVINO format.

    Args:
        model_id: HuggingFace model ID or local path
        output_dir: Output directory for converted models
        quantization_config: Optional quantization configuration for weight compression
        use_local_dir: If True, download model to local directory first
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model conversion. Please install torch.")

    thinker_output_dir = Path(output_dir) / "thinker"

    thinker_lang_path = thinker_output_dir / THINKER_LANGUAGE_NAME
    thinker_audio_path = thinker_output_dir / THINKER_AUDIO_NAME
    thinker_audio_encoder_path = thinker_output_dir / THINKER_AUDIO_ENCODER_NAME
    thinker_embedding_path = thinker_output_dir / THINKER_EMBEDDING_NAME

    if all(
        [
            thinker_lang_path.exists(),
            thinker_audio_path.exists(),
            thinker_audio_encoder_path.exists(),
            thinker_embedding_path.exists(),
        ]
    ):
        print(f"✅ {model_id} model already converted. You can find results in {output_dir}")
        return

    print(f"⌛ {model_id} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")

    if use_local_dir:
        ckpt = Path(output_dir) / "ckpt"
        if not ckpt.exists():
            snapshot_download(model_id, local_dir=ckpt, force_download=True)
    else:
        ckpt = model_id

    config = Qwen3ASRConfig.from_pretrained(ckpt)
    config.thinker_config.text_config._attn_implementation_autoset = False
    config.thinker_config.text_config._attn_implementation = "sdpa"

    model = Qwen3ASRForConditionalGeneration.from_pretrained(ckpt, config=config, torch_dtype=torch.float16)
    model.eval()

    # Try to load processor if available
    try:
        processor = AutoProcessor.from_pretrained(ckpt, fix_mistral_regex=True)
        processor.save_pretrained(output_dir)
    except Exception as e:
        print(f"⚠️ Could not load processor: {e}")

    config.save_pretrained(output_dir)
    print("✅ Original model successfully loaded")

    # Create output directories
    thinker_output_dir.mkdir(parents=True, exist_ok=True)

    # Convert thinker embedding model
    if not thinker_embedding_path.exists():
        print("⌛ Convert thinker embedding model")
        __make_16bit_traceable(model.thinker.model.get_input_embeddings())
        ov_model = ov.convert_model(
            model.thinker.model.get_input_embeddings(),
            example_input=torch.ones([2, 2], dtype=torch.int64),
        )
        ov.save_model(ov_model, thinker_embedding_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Thinker embedding model successfully converted")

    # Convert audio encoder model (Conv2D part)
    def forward_wrap_audio(self, padded_feature):
        """
        Audio encoder forward for Conv2D part.
        Input: padded_feature [batch, mel_bins, time]
        Output: padded_embed [batch, time_downsampled, d_model]
        """
        padded_embed = nn.functional.gelu(self.conv2d1(padded_feature.unsqueeze(1)))
        padded_embed = nn.functional.gelu(self.conv2d2(padded_embed))
        padded_embed = nn.functional.gelu(self.conv2d3(padded_embed))
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        return padded_embed

    def forward_wrap_audio_encoder(self, hidden_states, cu_seqlens):
        """
        Audio encoder forward for transformer layers.
        Input: hidden_states [seq_len, d_model], cu_seqlens [num_chunks + 1]
        Output: hidden_states [seq_len, output_dim]
        """
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states

    audio = model.thinker.audio_tower
    audio._orig_forward = audio.forward

    # Get dimensions from audio config
    num_mel_bins = audio.config.num_mel_bins
    d_model = audio.config.d_model

    if not thinker_audio_path.exists():
        print("⌛ Convert thinker audio model (Conv2D part)")
        __make_16bit_traceable(audio)
        audio.forward = types.MethodType(forward_wrap_audio, audio)
        ov_model = ov.convert_model(
            audio,
            example_input={
                "padded_feature": torch.randn([3, num_mel_bins, 100], dtype=torch.float32),
            },
            input=[ov.PartialShape([-1, num_mel_bins, -1])],
        )
        ov.save_model(ov_model, thinker_audio_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Thinker audio model (Conv2D part) successfully converted")

    if not thinker_audio_encoder_path.exists():
        print("⌛ Convert thinker audio encoder model (Transformer layers)")
        audio.forward = audio._orig_forward
        audio.forward = types.MethodType(forward_wrap_audio_encoder, audio)
        __make_16bit_traceable(audio)
        ov_model = ov.convert_model(
            audio,
            example_input={
                "hidden_states": torch.randn([5, d_model], dtype=torch.float32),
                "cu_seqlens": torch.tensor([0, 5], dtype=torch.int32),
            },
            input=[
                ov.PartialShape([-1, d_model]),
                ov.PartialShape([-1]),
            ],
        )
        ov.save_model(ov_model, thinker_audio_encoder_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Thinker audio encoder model (Transformer layers) successfully converted")

    # Convert Thinker Language model
    if not thinker_lang_path.exists():
        print("⌛ Convert Thinker Language model")

        def forward_wrap_thinker(
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
            cache_position: Optional[torch.LongTensor] = None,
        ):
            if past_key_values is not None:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

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
                cache_position=cache_position,
            )
            if past_key_values is not None:
                outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            output = (logits, outputs.past_key_values)

            return output

        lang_model = model.thinker
        hidden_size = lang_model.model.config.hidden_size
        patch_cos_sin_cached_fp32(lang_model)
        if hasattr(lang_model, "model"):
            patch_cos_sin_cached_fp32(lang_model.model)
        lang_model._orig_forward = lang_model.forward
        lang_model.forward = types.MethodType(forward_wrap_thinker, lang_model)

        num_pkv = lang_model.model.config.num_hidden_layers
        pkv_shape = (
            2,
            lang_model.model.config.num_key_value_heads,
            2,
            lang_model.model.config.head_dim,
        )

        cache_position = torch.arange(2, 4)
        position_ids = cache_position.view(1, 1, -1).expand(3, 2, -1)

        input_embeds = torch.randn((2, 2, hidden_size))
        attention_mask = torch.ones([2, 4], dtype=torch.long)
        input_names = ["attention_mask", "position_ids"]
        output_names = ["logits"]
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
                        lang_model.model.config.head_dim,
                    ]
                )
            ]
            * 2
            * num_pkv
        )
        input_shapes += [ov.PartialShape([-1, -1, hidden_size])]

        __make_16bit_traceable(lang_model)
        ov_model = ov.convert_model(lang_model, example_input=example_input, input=input_shapes)

        for input, input_name in zip(ov_model.inputs, input_names):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})

        patch_stateful(ov_model, 1)
        print("✅ Thinker language model successfully converted")

        if quantization_config is not None:
            if not NNCF_AVAILABLE:
                print("⚠️ NNCF is not available. Skipping weight compression.")
            else:
                print(f"⌛ Weights compression with {quantization_config['mode']} mode started")
                ov_model = nncf.compress_weights(ov_model, **quantization_config)
                print("✅ Weights compression finished")

        ov.save_model(ov_model, thinker_lang_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print(f"✅ Thinker model conversion finished. You can find results in {output_dir}")

    del model
    gc.collect()
    print(f"✅ {model_id} model conversion finished. You can find results in {output_dir}")


# ==================== Inference-only imports ====================
from dataclasses import dataclass
from typing import List, Union

# Import GenerationMixin for inference
try:
    from transformers.generation import GenerationMixin, GenerationConfig
    from transformers.modeling_outputs import BaseModelOutput, ModelOutput

    GENERATION_MIXIN_AVAILABLE = True
except ImportError:
    GENERATION_MIXIN_AVAILABLE = False

# Import inference utilities from Qwen3-ASR
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
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

    INFERENCE_UTILS_AVAILABLE = True
except ImportError:
    INFERENCE_UTILS_AVAILABLE = False
    SAMPLE_RATE = 16000
    MAX_ASR_INPUT_SECONDS = 1200
    SUPPORTED_LANGUAGES = ["Chinese", "English"]


@dataclass
class ASRTranscription:
    """
    One transcription result.

    Attributes:
        language (str): Merged language string for the sample.
        text (str): Transcribed text.
        time_stamps (Optional[Any]): Forced aligner output (not supported in OV version).
    """

    language: str
    text: str
    time_stamps: Optional[Any] = None


@dataclass
class Qwen3ASRThinkerCausalLMOutputWithPast(ModelOutput):
    """Output class for ASR Thinker model."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    rope_deltas: Optional[torch.LongTensor] = None


class SinusoidsPositionEmbedding:
    """Whisper-style sinusoidal positional embeddings for audio encoder.

    Matches the original Qwen3-ASR / Whisper formula exactly:
      inv_timescales[i] = exp(-log(max_timescale) / (channels//2 - 1) * i)
      pe = [sin(t * inv_timescales), cos(t * inv_timescales)]  (sins then cosines)
    """

    def __init__(self, length: int, channels: int, max_timescale: float = 10000.0):
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding requires even number of channels")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        # All sins first, then all cosines — matches original model layout
        self.positional_embedding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    def __getitem__(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]


class OVQwen3ASRThinkerForConditionalGeneration(GenerationMixin):
    """
    OpenVINO wrapper for Qwen3-ASR Thinker model with GenerationMixin support.
    This is the main ASR model that processes audio and generates text.
    """

    _is_stateful = False

    def __init__(self, model_dir, device, config):
        self.model_dir = Path(model_dir)
        self.config = config
        self.device_str = device
        self.device = torch.device("cpu")
        self.dtype = torch.float16

        # Load audio encoder components
        print(f"⌛ Loading audio conv model from {self.model_dir / THINKER_AUDIO_NAME}")
        self.audio_conv = ov.Core().compile_model(self.model_dir / THINKER_AUDIO_NAME, device)

        print(f"⌛ Loading audio encoder model from {self.model_dir / THINKER_AUDIO_ENCODER_NAME}")
        self.audio_encoder = ov.Core().compile_model(self.model_dir / THINKER_AUDIO_ENCODER_NAME, device)

        # Load embedding model
        print(f"⌛ Loading embedding model from {self.model_dir / THINKER_EMBEDDING_NAME}")
        self.embed_tokens_model = ov.Core().compile_model(self.model_dir / THINKER_EMBEDDING_NAME, device)

        # Load language model (stateful)
        print(f"⌛ Loading language model from {self.model_dir / THINKER_LANGUAGE_NAME}")
        self.model = ov.Core().read_model(self.model_dir / THINKER_LANGUAGE_NAME)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        compiled_model = ov.Core().compile_model(self.model, device)
        self.request = compiled_model.create_infer_request()

        # Create embedding wrapper
        self._embedding_wrapper = self._create_embedding_wrapper()
        self.get_input_embeddings = lambda: self._embedding_wrapper

        # Audio config
        audio_config = self.config.audio_config
        self.max_source_positions = audio_config.max_source_positions
        self.n_window = audio_config.n_window
        self.n_window_infer = getattr(audio_config, "n_window_infer", self.n_window * 2)
        embed_dim = audio_config.d_model

        # Positional embeddings for audio
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)

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

        # Token IDs
        self.audio_token_id = self.config.audio_token_id
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def _create_embedding_wrapper(self):
        """Create a callable wrapper for embeddings that works with OpenVINO."""

        def embedding_fn(input_ids):
            if isinstance(input_ids, torch.Tensor):
                input_np = input_ids.numpy()
            else:
                input_np = input_ids
            result = self.embed_tokens_model(input_np)[0]
            return torch.from_numpy(result)

        return embedding_fn

    def can_generate(self):
        """Returns True for GenerationMixin validation."""
        return True

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def audio_tower(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> BaseModelOutput:
        """
        Process audio through conv layers and transformer encoder.

        Args:
            input_features: Audio mel features [mel_bins, time]
            feature_lens: Length of audio features

        Returns:
            BaseModelOutput with audio embeddings
        """
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        # Build chunk lengths
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum().item(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        # Split and pad features
        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)

        # Get feature lengths after CNN
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )

        # Process through CNN
        padded_embed = torch.from_numpy(self.audio_conv(padded_feature.numpy())[0])

        # Add positional embeddings
        positional_embedding = self.positional_embedding[padded_embed.shape[1]].unsqueeze(0).to(padded_embed.dtype)
        padded_embed = padded_embed + positional_embedding

        # Extract valid hidden states
        hidden_states = padded_embed[padded_mask_after_cnn]

        # Build cu_seqlens for transformer
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len.item() // window_aftercnn)
            remainder = cnn_len.item() % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        # Process through encoder transformer
        hidden_states = torch.from_numpy(
            self.audio_encoder(
                {
                    "hidden_states": hidden_states.numpy().astype(np.float32),
                    "cu_seqlens": cu_seqlens.numpy(),
                }
            )[0]
        )

        return BaseModelOutput(last_hidden_state=hidden_states)

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features: Audio mel features [batch, mel_bins, time]
            feature_attention_mask: Mask for attention
            audio_feature_lengths: Length of each audio

        Returns:
            Audio embeddings
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)

        # Process each audio separately (following original model)
        audio_features = []
        for input_feature, feature_len in zip(input_features, feature_lens):
            audio_output = self.audio_tower(
                input_feature[:, :feature_len],
                feature_lens=feature_len.unsqueeze(0),
            )
            audio_feature = audio_output.last_hidden_state
            audio_features.append(audio_feature)

        audio_features = torch.cat(audio_features, dim=0)
        return audio_features

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        Obtains audio placeholder mask from input_ids or inputs_embeds.
        """
        if input_ids is None:
            special_audio_mask = (
                inputs_embeds == self.get_input_embeddings()(torch.tensor(self.audio_token_id, dtype=torch.long, device=inputs_embeds.device))
            ).all(-1)
        else:
            special_audio_mask = input_ids == self.audio_token_id

        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_audio_mask

    def get_rope_index(self, attention_mask: torch.Tensor):
        """Calculate mRoPE position IDs for the model."""
        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ) -> Qwen3ASRThinkerCausalLMOutputWithPast:
        """
        Forward pass through the ASR Thinker model.
        """
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Process audio features
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        # Handle feature attention mask
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        # Calculate position IDs
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(attention_mask)
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device if input_ids is not None else self.device)
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

        return Qwen3ASRThinkerCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=((),),
            hidden_states=None,
            attentions=None,
            rope_deltas=self.rope_deltas,
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
        input_features=None,
        feature_attention_mask=None,
        **kwargs,
    ):
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
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        # Only pass input_features on first step
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["input_features"] = None

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        self.next_beam_idx = np.array(beam_idx)
        return past_key_values

    def _get_past_length(self, past_key_values=None):
        return self._past_length if past_key_values else 0


class OVQwen3ASRModel:
    """
    OpenVINO-based Qwen3-ASR model for inference.
    Provides the same API as Qwen3ASRModel.transcribe().

    This class follows the original Qwen3ASRForConditionalGeneration structure:
    - Uses OVQwen3ASRThinkerForConditionalGeneration as self.thinker
    - generate() calls self.thinker.generate()
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "CPU",
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 512,
    ):
        """
        Initialize the OpenVINO Qwen3-ASR model.

        Args:
            model_dir: Directory containing the converted OpenVINO models
            device: Device to run inference on (e.g., "CPU", "GPU", "NPU")
            max_inference_batch_size: Batch size limit for inference
            max_new_tokens: Maximum number of tokens to generate
        """
        if not INFERENCE_UTILS_AVAILABLE:
            raise ImportError("Qwen3-ASR inference utilities not available. Please install qwen_asr package.")

        if not GENERATION_MIXIN_AVAILABLE:
            raise ImportError("GenerationMixin not available. Please install transformers>=4.40.")

        self.model_dir = Path(model_dir)
        self.device = device
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens

        # Load config
        self.config = Qwen3ASRConfig.from_pretrained(model_dir)

        # Initialize thinker using GenerationMixin wrapper
        thinker_dir = self.model_dir / "thinker"
        self.thinker = OVQwen3ASRThinkerForConditionalGeneration(thinker_dir, device, self.config.thinker_config)

        # Load processor
        try:
            self.processor = Qwen3ASRProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
            print("✅ Processor loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load processor: {e}")
            self.processor = None

        print("✅ OVQwen3ASRModel initialized successfully")

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "CPU",
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> "OVQwen3ASRModel":
        """
        Load OpenVINO Qwen3-ASR model from a directory.

        Args:
            model_dir: Directory containing the converted OpenVINO models
            device: Device to run inference on
            max_inference_batch_size: Batch size limit for inference
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            OVQwen3ASRModel instance
        """
        return cls(
            model_dir=model_dir,
            device=device,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )

    def get_support_languages(self) -> List[str]:
        """Returns the supported language list (same as original model API)."""
        return self.config.support_languages if hasattr(self.config, "support_languages") else list(SUPPORTED_LANGUAGES)

    def get_supported_languages(self) -> List[str]:
        """Returns the supported language list."""
        return self.get_support_languages()

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 4096,
        eos_token_id: Union[int, List[int]] = [151645, 151643],
        **kwargs,
    ):
        """
        Generate text from audio input using the thinker model.

        This method follows the original Qwen3ASRForConditionalGeneration.generate() exactly.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum new tokens to generate
            eos_token_id: EOS token ID(s)
            **kwargs: Additional arguments passed to thinker.generate()

        Returns:
            Generation output from thinker
        """
        shared_kwargs = {}
        thinker_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_token_id,
        }

        for key, value in kwargs.items():
            # Process special input values
            if key == "feature_attention_mask":
                thinker_kwargs[key] = value
            elif key in ("input_features", "attention_mask"):
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value

        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value

        thinker_result = self.thinker.generate(input_ids=input_ids, return_dict_in_generate=True, **thinker_kwargs)

        return thinker_result

    def _build_messages(self, context: str, audio_payload: Any) -> List[dict]:
        """Build messages for chat template."""
        return [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
        ]

    def _build_text_prompt(self, context: str, force_language: Optional[str]) -> str:
        """
        Build the string prompt for one request.
        """
        msgs = self._build_messages(context=context, audio_payload="")
        base = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if force_language:
            base = base + f"language {force_language}{'<asr_text>'}"
        return base

    def _infer_asr(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        """
        Run ASR inference for chunk-level items using self.generate().

        Args:
            contexts: List of context strings.
            wavs: List of mono waveforms (np.ndarray).
            languages: List of forced languages or None.

        Returns:
            List[str]: Raw decoded strings (one per chunk).
        """
        outs: List[str] = []

        texts = [self._build_text_prompt(context=c, force_language=fl) for c, fl in zip(contexts, languages)]

        batch_size = self.max_inference_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(texts)

        for i in range(0, len(texts), batch_size):
            sub_text = texts[i : i + batch_size]
            sub_wavs = wavs[i : i + batch_size]

            # Process inputs using processor
            inputs = self.processor(text=sub_text, audio=sub_wavs, return_tensors="pt", padding=True)

            # Get input tensors
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            input_features = inputs["input_features"]
            feature_attention_mask = inputs["feature_attention_mask"]

            # Reset thinker state
            self.thinker.rope_deltas = None

            # Generate using self.generate() which calls self.thinker.generate()
            generation_output = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                max_new_tokens=self.max_new_tokens,
            )

            # Extract generated sequences
            generated_ids = generation_output.sequences

            # Decode
            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outs.extend(decoded)

        return outs

    def transcribe(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[ASRTranscription]:
        """
        Transcribe audio with optional context.

        Args:
            audio: Audio input(s). Supported:
                - str: local path / URL / base64 data url
                - (np.ndarray, sr): waveform and sample rate tuple
                - list of above
            context: Context string(s). If scalar, broadcast to batch size.
            language: Optional language(s). If provided, force output to that language.
            return_time_stamps: Not supported in OpenVINO version.

        Returns:
            List[ASRTranscription]: One result per input audio.
        """
        if return_time_stamps:
            raise ValueError("return_time_stamps is not supported in OpenVINO version")

        # Normalize audio inputs
        wavs = normalize_audios(audio)
        n = len(wavs)

        # Normalize contexts
        ctxs = context if isinstance(context, list) else [context]
        if len(ctxs) == 1 and n > 1:
            ctxs = ctxs * n
        if len(ctxs) != n:
            raise ValueError(f"Batch size mismatch: audio={n}, context={len(ctxs)}")

        # Normalize languages
        langs_in: List[Optional[str]]
        if language is None:
            langs_in = [None] * n
        else:
            langs_in = language if isinstance(language, list) else [language]
            if len(langs_in) == 1 and n > 1:
                langs_in = langs_in * n
            if len(langs_in) != n:
                raise ValueError(f"Batch size mismatch: audio={n}, language={len(langs_in)}")

        langs_norm: List[Optional[str]] = []
        for l in langs_in:
            if l is None or str(l).strip() == "":
                langs_norm.append(None)
            else:
                ln = normalize_language_name(str(l))
                validate_language(ln)
                langs_norm.append(ln)

        max_chunk_sec = MAX_ASR_INPUT_SECONDS

        # Chunk audios and record mapping
        chunks: List[AudioChunk] = []
        for i, wav in enumerate(wavs):
            parts = split_audio_into_chunks(
                wav=wav,
                sr=SAMPLE_RATE,
                max_chunk_sec=max_chunk_sec,
            )
            for j, (cwav, offset_sec) in enumerate(parts):
                chunks.append(AudioChunk(orig_index=i, chunk_index=j, wav=cwav, sr=SAMPLE_RATE, offset_sec=offset_sec))

        # Run ASR on chunks
        chunk_ctx: List[str] = [ctxs[c.orig_index] for c in chunks]
        chunk_lang: List[Optional[str]] = [langs_norm[c.orig_index] for c in chunks]
        chunk_wavs: List[np.ndarray] = [c.wav for c in chunks]
        raw_outputs = self._infer_asr(chunk_ctx, chunk_wavs, chunk_lang)

        # Parse outputs
        per_chunk_lang: List[str] = []
        per_chunk_text: List[str] = []
        for out, forced_lang in zip(raw_outputs, chunk_lang):
            lang, txt = parse_asr_output(out, user_language=forced_lang)
            per_chunk_lang.append(lang)
            per_chunk_text.append(txt)

        # Merge chunks back to original samples
        out_langs: List[List[str]] = [[] for _ in range(n)]
        out_texts: List[List[str]] = [[] for _ in range(n)]

        for c, lang, txt in zip(chunks, per_chunk_lang, per_chunk_text):
            out_langs[c.orig_index].append(lang)
            out_texts[c.orig_index].append(txt)

        results: List[ASRTranscription] = []
        for i in range(n):
            merged_text = "".join([t for t in out_texts[i] if t is not None])
            merged_language = merge_languages(out_langs[i])
            results.append(ASRTranscription(language=merged_language, text=merged_text, time_stamps=None))

        return results
