from pathlib import Path
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoConfig, GenerationConfig, TextStreamer
from transformers.cache_utils import DynamicCache, Cache
from transformers.models.llama.modeling_llama import repeat_kv
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
from typing import Optional, Union, List, Tuple, Dict
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput, BaseModelOutput
import openvino.runtime.opset13 as ops
import types
import openvino as ov
import gc
import torch
import numpy as np
from dataclasses import dataclass
import requests
from PIL import Image
from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher
import time

IMAGE_ENCODER = "openvino_vision_encoder.xml"
LANGUAGE_MODEL = "openvino_language_model.xml"


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def _add_runtime_options_to_rt_info(model: ov.Model, options: Dict):
    """
    Add runtime optinos
    """
    try:
        for name, value in options.items():
            model.set_rt_info(value, ["runtime_options", name])
    except Exception:
        pass

    return model


class InsertSlice(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Result")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            if root is None:
                return False
            if len(root.get_output_partial_shape(0)) == 3:
                parent = root.input_value(0).get_node()
                grand_parent = parent.input_value(0).get_node()

                grand_parent_output = parent.input(0).get_source_output()
                consumers = grand_parent_output.get_target_inputs()
                start = np.array([0, -1, 0], dtype=np.int32)
                stop = np.array([1, -2, grand_parent_output.get_partial_shape()[-1].get_length()], dtype=np.int32)
                step = np.array([1, -1, 1], dtype=np.int32)
                axes = np.array([0, 1, 2], dtype=np.int32)
                slice = ops.slice(grand_parent, start, stop, step, axes, name="inserted_slice")
                for consumer in consumers:
                    consumer.replace_source_output(slice.output(0))
                self.model_changed = True
                # Use new operation for additional matching
                self.register_new_node(slice)
                print("applied slice for lm head")

                return True

        self.register_matcher(Matcher(param, "InsertSlice"), callback)


STR_TO_OV_TYPE = {
    "boolean": ov.Type.boolean,
    "f16": ov.Type.f16,
    "f32": ov.Type.f32,
    "f64": ov.Type.f64,
    "i8": ov.Type.i8,
    "i16": ov.Type.i16,
    "i32": ov.Type.i32,
    "i64": ov.Type.i64,
    "u8": ov.Type.u8,
    "u16": ov.Type.u16,
    "u32": ov.Type.u32,
    "u64": ov.Type.u64,
    "bf16": ov.Type.bf16,
}


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
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
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
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("input_ids").get_partial_shape()[0]
    beam_idx = ops.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = ops.gather(parameter_output_port, beam_idx, ops.constant(gather_dim))
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
    input_ids = ov_model.input("input_ids")
    batch = ops.gather(
        ops.shape_of(input_ids, output_type="i64"),
        ops.constant([0]),
        ops.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(ops.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = ops.concat(dims, axis=0)
            broadcast = ops.broadcast(ops.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
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


def patch_stateful(ov_model):
    key_value_input_names = [key_name for key in ov_model.inputs for key_name in key.get_names() if "past_key_values" in key_name]
    key_value_output_names = [key_name for key in ov_model.outputs for key_name in key.get_names() if "present" in key_name]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
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


def convert_mllama(model_id, out_dir):
    out_dir = Path(out_dir)

    img_encoder_path = out_dir / IMAGE_ENCODER
    lang_model_path = out_dir / LANGUAGE_MODEL

    requires_conversion = not all([img_encoder_path.exists(), lang_model_path.exists()])
    if not requires_conversion:
        print(f"✅ Model already converted and can be found in {out_dir}")
        return
    print("⌛ Load original model")
    model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
    model.eval()
    model.config.save_pretrained(out_dir)
    model.generation_config.save_pretrained(out_dir)
    __make_16bit_traceable(model)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.save_pretrained(out_dir)

    if not img_encoder_path.exists():
        print("⌛ Convert vision model...")

        def _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask: torch.Tensor,
            num_patches: int,
            target_length: int,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            # Expand aspect ratio mask to target_length
            batch_size, max_num_tiles = aspect_ratio_mask.shape
            attention_mask = aspect_ratio_mask.view(batch_size, max_num_tiles, 1, 1).to(dtype)
            attention_mask = attention_mask.repeat(1, 1, target_length, 1)

            # Mask padding patches
            pad_patches = target_length - num_patches
            attention_mask[:, :, -pad_patches:] = 0

            # Invert the mask (0 -> 1, 1 -> 0)
            attention_mask = 1 - attention_mask

            # Reshape to 2D and create 4D attention mask
            # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
            attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
            attention_mask = attention_mask @ attention_mask.transpose(-1, -2) * torch.finfo(torch.float16).min
            attention_mask = attention_mask.unsqueeze(1)

            return attention_mask

        def _img_encoder_forward(
            self,
            pixel_values: torch.Tensor,
            aspect_ratio_ids: torch.Tensor,
            aspect_ratio_mask: torch.Tensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

            pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
            aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)

            # Patch embedding
            patch_embeds = self.patch_embedding(pixel_values.to(self.dtype).to(self.device))
            hidden_state = patch_embeds.flatten(2).transpose(1, 2)

            # Tile embeddings
            _, num_patches, dim = hidden_state.shape
            hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
            hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)

            # Add cls token
            hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
            hidden_state = self.apply_class_embedding(hidden_state)
            num_patches += 1

            # Position embeddings
            hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
            hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

            hidden_state = self.layernorm_pre(hidden_state)

            # Compute the number of tokens to pad
            num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
            # Compute padding tuple for pad function
            padding = (0, 0, 0, num_padding_patches)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
            # Pad the tensor
            hidden_state = torch.nn.functional.pad(hidden_state, padding, mode="constant", value=0)
            slice_index = -num_padding_patches if num_padding_patches > 0 else None

            # Prepare attention mask
            attention_mask = aspect_ratio_mask.reshape(batch_size * num_concurrent_media, -1)
            attention_mask = _prepare_aspect_ratio_attention_mask(
                aspect_ratio_mask=attention_mask,
                num_patches=self.num_patches,
                target_length=hidden_state.shape[2],
                dtype=self.dtype,
            )

            # Apply encoder
            hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
            output = self.transformer(
                hidden_state,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=output_attentions,
            )
            hidden_state = output[0]

            hidden_state = self.layernorm_post(hidden_state)

            # Apply global encoder
            hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim)
            hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
            hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles * (num_patches + num_padding_patches), dim)
            global_output = self.global_transformer(
                hidden_state,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            hidden_state = global_output[0]

            # Remove padding form hidden state
            hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim)
            hidden_state = hidden_state[:, :, :slice_index]
            hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, dim)

            # Collect intermediate layer outputs from encoder output
            all_intermediate_hidden_states = [output[1][i] for i in self.intermediate_layers_indices]
            intermediate_hidden_states = torch.stack(all_intermediate_hidden_states, dim=-1)

            # Remove padding from intermediate hidden states
            intermediate_hidden_states = intermediate_hidden_states.reshape(batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1)
            intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
            intermediate_hidden_states = intermediate_hidden_states.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, -1)

            # Concatenate final hidden state and intermediate hidden states
            hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)

            if output_hidden_states:
                hidden_states = tuple(all_intermediate_hidden_states) + tuple(global_output[1])
            else:
                hidden_states = None

            if output_attentions:
                # global transformer in contrast to `self.transformer` doesn't always return hidden states so we might go index out-of-range
                global_attn = tuple(global_output[2]) if output_hidden_states else tuple(global_output[1])
                attentions = tuple(output[2]) + global_attn
            else:
                attentions = None

            if not return_dict:
                return tuple(v for v in [hidden_state, hidden_states, attentions] if v is not None)

            return BaseModelOutput(
                last_hidden_state=hidden_state,
                hidden_states=hidden_states,
                attentions=attentions,
            )

        class VisionEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.model.vision_model.forward = types.MethodType(_img_encoder_forward, self.model.vision_model)

            def forward(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask):
                bsz = pixel_values.shape[0]
                cross_attention_states = self.model.vision_model(pixel_values, aspect_ratio_ids, aspect_ratio_mask)[0]
                cross_attention_states = self.model.multi_modal_projector(cross_attention_states).reshape(
                    -1, cross_attention_states.shape[-2], self.model.hidden_size
                )
                cross_attention_kv_cache = ()
                for layer_idx in self.model.language_model.model.cross_attention_layers:
                    layer = self.model.language_model.model.layers[layer_idx]
                    cross_attn = layer.cross_attn
                    key_states = cross_attn.k_proj(cross_attention_states)
                    value_states = cross_attn.v_proj(cross_attention_states)
                    key_states = key_states.view(bsz, -1, cross_attn.num_key_value_heads, cross_attn.head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, -1, cross_attn.num_key_value_heads, cross_attn.head_dim).transpose(1, 2)
                    key_states = repeat_kv(key_states, cross_attn.num_key_value_groups)
                    value_states = repeat_kv(value_states, cross_attn.num_key_value_groups)

                    key_states = cross_attn.k_norm(key_states)
                    cross_attention_kv_cache += ((key_states, value_states),)

                return cross_attention_states, cross_attention_kv_cache

        image_encoder = VisionEncoder(model)
        image_encoder.eval()

        with torch.no_grad():
            ov_model = ov.convert_model(
                image_encoder,
                example_input={
                    "pixel_values": torch.randn((1, 1, 4, 3, model.config.vision_config.image_size, model.config.vision_config.image_size)),
                    "aspect_ratio_ids": torch.tensor([[6]]),
                    "aspect_ratio_mask": torch.tensor([[[1, 1, 1, 1]]]),
                },
            )

        output_names = ["cross_attention_states"]

        for i in model.config.text_config.cross_attention_layers:
            output_names.extend([f"cross_attn_key_values.{i}.key", f"cross_attn_key_values.{i}.value"])

        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})

        _add_runtime_options_to_rt_info(ov_model, {"ACTIVATIONS_SCALE_FACTOR": "8.0", "KV_CACHE_PRECISION": "f16"})

        ov.save_model(ov_model, img_encoder_path)
        del ov_model
        cleanup_torchscript_cache()
        del image_encoder
        gc.collect()

        print("✅ Vision model successfully converted")

    if not lang_model_path.exists():
        print("⌛ Convert language model...")

        def lm_forward_wrapper(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cross_attention_mask: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            cross_attn_key_values: Optional[List[torch.FloatTensor]] = None,
        ):
            common_cache = []
            self_cache_id = 0
            cross_cache_id = 0

            for i in range(len(self.model.layers)):
                if i in self.model.cross_attention_layers:
                    common_cache.append(cross_attn_key_values[cross_cache_id])
                    cross_cache_id += 1
                else:
                    common_cache.append(past_key_values[self_cache_id])
                    self_cache_id += 1

            common_cache = DynamicCache.from_legacy_cache(common_cache)
            result = self.orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cross_attention_mask=cross_attention_mask,
                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                past_key_values=common_cache,
                cache_position=cache_position,
            )
            present_kv = [kv_cache for idx, kv_cache in enumerate(result.past_key_values.to_legacy_cache()) if idx not in self.model.cross_attention_layers]
            return result.logits, tuple(present_kv)

        def cross_attn_forward(
            self,
            hidden_states: torch.Tensor,
            cross_attention_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Cache] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            use_cache: bool = None,
            cache_position: Optional[torch.LongTensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            """Input shape: Batch x Time x Channel"""
            bsz, q_len, _ = hidden_states.size()
            query_states = self.q_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            query_states = self.q_norm(query_states)

            if cross_attention_states is not None:
                key_states = self.k_proj(cross_attention_states)
                value_states = self.v_proj(cross_attention_states)
                key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                key_states = self.k_norm(key_states)
                if past_key_value is not None:
                    # if we have a new image + new tokens, we only computed key_states on that new image
                    # we still update the cross key states, past_image, new_image. And use it!
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, {"cache_position": cache_position})
            elif past_key_value.get_seq_length(self.layer_idx) != 0:
                key_states, value_states = (
                    past_key_value.key_cache[self.layer_idx],
                    past_key_value.value_cache[self.layer_idx],
                )
            else:
                raise ValueError("Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!")

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=causal_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        def _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask: torch.Tensor,
            sequence_length: int,
            target_length: int,
            dtype: torch.dtype,
            device: torch.device,
            min_dtype: float,
            cache_position: torch.Tensor,
            batch_size: int,
        ):
            """
            Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
            `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

            Args:
                attention_mask (`torch.Tensor`):
                    A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
                sequence_length (`int`):
                    The sequence length being processed.
                target_length (`int`):
                    The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
                dtype (`torch.dtype`):
                    The dtype to use for the 4D attention mask.
                device (`torch.device`):
                    The device to plcae the 4D attention mask on.
                min_dtype (`float`):
                    The minimum value representable with the dtype `dtype`.
                cache_position (`torch.Tensor`):
                    Indices depicting the position of the input sequence tokens in the sequence.
                batch_size (`torch.Tensor`):
                    Batch size.
            """
            if attention_mask is not None and attention_mask.dim() == 4:
                # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
                causal_mask = attention_mask
            else:
                causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
                if sequence_length != 1:
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
                causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
                if attention_mask is not None:
                    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                    mask_length = attention_mask.shape[-1]
                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

            return causal_mask

        def _lm_update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
        ):

            from transformers.modeling_attn_mask_utils import AttentionMaskConverter

            if self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None

            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
            # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
            # to infer the attention mask.
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

            # When output attention is True, sdpa implementation's forward method calls the eager implementation's forward
            # TODO: we have only SDPA currently and there's a bug when attn-bias is passed. Need to add eager attn and return the line
            # self.config._attn_implementation == "sdpa" and
            if self.config._attn_implementation == "sdpa" and not output_attentions:
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
                ):
                    return None

            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = torch.finfo(torch.float16).min
            sequence_length = input_tensor.shape[1]
            target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1

            # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
            causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=target_length,
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=input_tensor.shape[0],
            )

            if self.config._attn_implementation == "sdpa" and attention_mask is not None and attention_mask.device.type == "cuda" and not output_attentions:
                # Attend to all tokens in fully masked rows in the causal_mask, for example, the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

            return causal_mask

        for layer_idx in model.language_model.model.cross_attention_layers:
            cross_attn = model.language_model.model.layers[layer_idx].cross_attn
            cross_attn.forward = types.MethodType(cross_attn_forward, cross_attn)

        model.language_model.orig_forward = model.language_model.forward
        model.language_model.forward = types.MethodType(lm_forward_wrapper, model.language_model)
        model.language_model.model._update_causal_mask = types.MethodType(_lm_update_causal_mask, model.language_model.model)
        example_inpit = {
            "input_ids": torch.ones([1, 2], dtype=torch.int64),
            "attention_mask": torch.ones([1, 4], dtype=torch.int64),
            "position_ids": torch.tensor([[2, 3]], dtype=torch.int64),
            "cross_attention_mask": torch.zeros([1, 1, 2, 6404]),
            "cache_position": torch.tensor([2, 3], dtype=torch.int64),
            "full_text_row_masked_out_mask": torch.ones([1, 1, 2, 1]),
        }

        input_names = list(example_inpit.keys())
        output_names = ["logits"]
        past_key_values = []
        cross_attn_key_values = []
        cross_attn_names = []
        pkv_in_names = []
        pkv_out_names = []

        for i in range(model.config.text_config.num_hidden_layers):
            if i in model.config.text_config.cross_attention_layers:
                cross_attn_key_values.append((torch.randn([1, 32, 6404, 128]), torch.randn(1, 32, 6404, 128)))
                cross_attn_names.extend([f"cross_attn_key_values.{i}.key", f"cross_attn_key_values.{i}.value"])
            else:
                past_key_values.append((torch.randn([1, 8, 2, 128]), torch.randn([1, 8, 2, 128])))
                pkv_in_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
                pkv_out_names.extend([f"present.{i}.key", f"present.{i}.value"])

        input_names.extend(pkv_in_names)
        output_names.extend(pkv_out_names)
        input_names.extend(cross_attn_names)

        example_inpit["past_key_values"] = past_key_values
        example_inpit["cross_attn_key_values"] = cross_attn_key_values
        model.language_model.eval()

        with torch.no_grad():
            ov_model = ov.convert_model(model.language_model, example_input=example_inpit)

        for input, input_name in zip(ov_model.inputs, input_names):
            input.get_tensor().set_names({input_name})
            if input_name.startswith("cross_attn_key_values"):
                input.get_node().set_partial_shape(ov.PartialShape([-1, 32, 6404, 128]))

        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})

        ov_model.validate_nodes_and_infer_types()
        patch_stateful(ov_model)
        _add_runtime_options_to_rt_info(ov_model, {"ACTIVATIONS_SCALE_FACTOR": "8.0", "KV_CACHE_PRECISION": "f16"})
        ov.save_model(ov_model, lang_model_path)
        del ov_model
        cleanup_torchscript_cache()
        del model
        gc.collect()
        print("✅ Language model successfully converted")
    print(f"✅ Model successfully converted and can be found in {out_dir}")


core = ov.Core()


@dataclass
class MLlamaOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attn_key_values: Optional[List[torch.FloatTensor]] = None


class OVMLlamaForConditionalGeneration(GenerationMixin):
    def __init__(
        self,
        model_dir: Union[str, Path],
        device: str = "CPU",
        ov_config: Optional[Dict[str, str]] = None,
        language_model_name=None,
        image_encoder_name=None,
        slice_lm_head=True,
        use_remote_tensors=True,
        dynamic_shape=False,
        force_fp32_pkv=False,
    ):
        model_dir = Path(model_dir)
        self.config = AutoConfig.from_pretrained(model_dir)
        self.generation_config = GenerationConfig.from_pretrained(model_dir)
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self._device = device
        self.ov_config = ov_config
        self.num_pkv = 2
        self._supports_cache_class = False
        self.next_beam_idx = None
        self._past_length = None
        if language_model_name:
            self.model = core.read_model(model_dir / language_model_name)
        else:
            self.model = core.read_model(model_dir / LANGUAGE_MODEL)
        if image_encoder_name:
            self.vision_model = core.read_model(model_dir / image_encoder_name)
        else:
            self.vision_model = core.read_model(model_dir / IMAGE_ENCODER)
        if not dynamic_shape:
            self.reshape_vision_model()
        self.update_pkv_precision(force_fp32=force_fp32_pkv)
        if slice_lm_head:
            self.slice_lm_head()
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.lm_cross_attn_inputs = [key for key in self.input_names if "cross_attn_key_values" in key]
        compiled_model = core.compile_model(self.model, device, ov_config)
        self.request = compiled_model.create_infer_request()
        self.cross_attn_outputs = [key.get_any_name() for key in self.vision_model.outputs if "cross_attn_key_values" in key.get_any_name()]
        compiled_vision_model = core.compile_model(self.vision_model, device, ov_config)
        self.vision_request = compiled_vision_model.create_infer_request()
        self.use_remote_tensors = use_remote_tensors and self._device == "GPU"
        if self.use_remote_tensors:
            self.prepare_remote_tensors()
        self.next_beam_idx = None
        self.num_patches = (self.config.vision_config.image_size // self.config.vision_config.patch_size) ** 2 + 1
        self._past_length = 0
        self.llm_infer_time = []
        self.vision_encoder_infer_time = []

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        return self._past_length

    def reshape_vision_model(self):
        self.vision_model.reshape(
            {
                0: ov.PartialShape([1, 1, 4, 3, self.config.vision_config.image_size, self.config.vision_config.image_size]),
                1: ov.PartialShape([1, 1]),
                2: ov.PartialShape([1, 1, 4]),
            }
        )

    def update_pkv_precision(self, force_fp32=False):
        pkv_precision = ov.Type.f32
        if not force_fp32:
            device = self._device.upper()
            try:
                if "INFERENCE_PRECISION_HINT" in core.get_property(device, "SUPPORTED_PROPERTIES"):
                    pkv_precision = core.get_property(device, "INFERENCE_PRECISION_HINT")
            except RuntimeError:  # use default precision when get_property fails, e.g. when device is "AUTO:GPU"
                pass

            # ov_config["INFERENCE_PRECISION_HINT"] may override the prefer precision
            if self.ov_config:
                inference_precision_hint = self.ov_config.get("INFERENCE_PRECISION_HINT", "")
                if inference_precision_hint in STR_TO_OV_TYPE:
                    pkv_precision = STR_TO_OV_TYPE[inference_precision_hint]

            ppp = ov.preprocess.PrePostProcessor(self.model)
            for key in self.model.inputs:
                if "cross_attn_key_values" in key.get_any_name() and pkv_precision != key.get_element_type():
                    ppp.input(key.get_any_name()).tensor().set_element_type(pkv_precision)

            self.model = ppp.build()

            ppp_v = ov.preprocess.PrePostProcessor(self.vision_model)
            for key in self.vision_model.outputs:
                if "cross_attn_key_values" in key.get_any_name() and pkv_precision != key.get_element_type():
                    ppp_v.output(key.get_any_name()).tensor().set_element_type(pkv_precision)
            self.vision_model = ppp_v.build()
            self._pkv_precision = pkv_precision

    def slice_lm_head(self):
        manager = Manager()
        manager.register_pass(InsertSlice())
        manager.run_passes(self.model)
        self.model.validate_nodes_and_infer_types()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[List[List[int]]] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[List[List[List[int]]]] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cross_attn_key_values: Optional[List[torch.Tensor]] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, MLlamaOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaForConditionalGeneration

        >>> model = MllamaForConditionalGeneration.from_pretrained("<mllama-checkpoint>")
        >>> processor = AutoProcessor.from_pretrained("<mllama-checkpoint>")

        >>> prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputCausalLMOutputWithPass, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "TODO: fill this out"
        ```"""

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one")

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            cross_attn_key_values = self.visual_encoder(pixel_values, aspect_ratio_ids, aspect_ratio_mask)
        cross_attention_mask, full_text_row_masked_out_mask = self._prepare_cross_attention_mask(
            cross_attention_mask,
            past_key_values=past_key_values,
            num_vision_tokens=self.num_patches,
            cross_attention_layers=cross_attn_key_values if past_key_values is not None else None,
            cross_attention_states=((),),
            device=self.device,
            dtype=torch.float32,
        )

        if cross_attention_mask is not None and cache_position is not None:
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]

        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            cross_attention_key_values=cross_attn_key_values,
        )

    def language_model(
        self,
        input_ids,
        attention_mask,
        position_ids,
        cross_attention_mask,
        full_text_row_masked_out_mask,
        past_key_values,
        cache_position,
        cross_attention_key_values,
    ):
        model_inputs = {
            "input_ids": ov.Tensor(np.array(input_ids)),
            "attention_mask": ov.Tensor(np.array(attention_mask)),
            "position_ids": ov.Tensor(np.array(position_ids)),
            "cross_attention_mask": ov.Tensor(np.array(cross_attention_mask)),
            "full_text_row_masked_out_mask": ov.Tensor(np.array(full_text_row_masked_out_mask)),
            "cache_position": ov.Tensor(np.array(cache_position)),
        }

        if past_key_values is None:
            self.request.reset_state()
            self.next_beam_idx = np.arange(input_ids.shape[0], dtype=int)
            self._past_length = 0
            self.llm_infer_time = []

        if not self.use_remote_tensors:
            model_inputs.update(dict(zip(self.lm_cross_attn_inputs, cross_attention_key_values)))
        if "beam_idx" in self.input_names:
            model_inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(input_ids.shape[0], dtype=int)

        start = time.perf_counter()
        self.request.start_async(model_inputs, share_inputs=True)
        self.request.wait()
        end = time.perf_counter()
        self.llm_infer_time.append(end - start)
        logits = torch.from_numpy(self.request.get_tensor("logits").data)
        past_key_values = ((),)
        self._past_length += input_ids.shape[1]
        out = MLlamaOutputWithPast(logits=logits, past_key_values=past_key_values, cross_attn_key_values=cross_attention_key_values)
        return out

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def __call__(self, *args, **kwargs) -> MLlamaOutputWithPast:
        return self.forward(
            *args,
            **kwargs,
        )

    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        cross_attn_key_values=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # The clone here is for the same reason as for `position_ids`.
        model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cross_attention_mask": cross_attention_mask,
                "cross_attn_key_values": cross_attn_key_values,
            }
        )

        # If we're in pre-fill or cacheless decoding step, then we need pixel_values and aspect ratios
        # to compute image hidden states, otherwise they are cache/home/ea/llama3.2/Llama-3.2-11B-Vision-Early/OVd within each cross attn layer
        if (input_ids == self.config.image_token_index).any():
            model_inputs["pixel_values"] = pixel_values
            model_inputs["aspect_ratio_ids"] = aspect_ratio_ids
            model_inputs["aspect_ratio_mask"] = aspect_ratio_mask

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        # add cross-attn mask for new token
        if cross_attention_mask_prev is not None:
            model_kwargs["cross_attention_mask"] = torch.cat([cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1)
        model_kwargs["cross_attn_key_values"] = outputs.cross_attn_key_values
        return model_kwargs

    def _prepare_cross_attention_mask(
        self,
        cross_attention_mask: torch.Tensor,
        past_key_values: Tuple,
        num_vision_tokens: int,
        cross_attention_states: torch.Tensor,
        cross_attention_layers: List[int],
        device: str,
        dtype: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cross_attention_mask is None:
            # should we raise error or prepare a full attn mask with all ones?
            return None, None
        else:
            # reshape so it can be used by attn module
            batch_size, text_total_length, *_ = cross_attention_mask.shape
            cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
            cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
            cross_attention_mask = cross_attention_mask.unsqueeze(1)

        # invert the mask
        inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
        cross_attention_mask = inverted_cross_attn_mask.masked_fill(inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min)

        # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
        # last dimension contains negative infinity values, otherwise it's 1
        negative_inf_value = torch.finfo(dtype).min
        full_text_row_masked_out_mask = (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
        cross_attention_mask *= full_text_row_masked_out_mask

        # In case we receive a new image but already have previous cross-attention key/values in cache,
        # then we need to extend the attention-mask and add previous images' lengths
        if past_key_values is not None and cross_attention_states is not None and cross_attention_layers is not None:
            # make all zeros mask for cross-attn-mask from previuos cached hidden_states, all zeros right?
            # i.e. extend current cross-attn-mask on image-seq-length dimension to account for past_seen_tokens
            past_cross_attn_kv_length = cross_attention_layers[0].shape[-2]
            past_cross_attn_mask = torch.zeros((*cross_attention_mask.shape[:-1], past_cross_attn_kv_length), dtype=dtype, device=device)
            # concatenate both on image-seq-length dimension
            cross_attention_mask = torch.cat([past_cross_attn_mask, cross_attention_mask], dim=-1)

        return cross_attention_mask, full_text_row_masked_out_mask

    def visual_encoder(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask):
        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            self.vision_encoder_infer_time = []
            start = time.perf_counter()
            # get vision tokens from vision model
            self.vision_request.start_async([pixel_values, aspect_ratio_ids, aspect_ratio_mask], share_inputs=True)
            self.vision_request.wait()
            end = time.perf_counter()
            cross_attn_key_values = [self.vision_request.get_tensor(name) for name in self.cross_attn_outputs]
            self.vision_encoder_infer_time.append(end - start)
        return cross_attn_key_values

    def prepare_vision_outputs(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask, cross_attention_mask=None, past_key_values=None, cache_position=None):
        cross_attn_key_values = self.visual_encoder(pixel_values, aspect_ratio_ids, aspect_ratio_mask)
        cross_attn_key_values = [v.data for v in cross_attn_key_values]
        cross_attention_mask, full_text_row_masked_out_mask = self._prepare_cross_attention_mask(
            cross_attention_mask,
            past_key_values=past_key_values,
            num_vision_tokens=self.num_patches,
            cross_attention_layers=cross_attn_key_values if past_key_values is not None else None,
            cross_attention_states=1,
            device=self.device,
            dtype=torch.float32,
        )

        if cross_attention_mask is not None and cache_position is not None:
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]

        return {
            "cross_attention_mask": cross_attention_mask,
            "full_text_row_masked_out_mask": full_text_row_masked_out_mask,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "cross_attention_key_values": cross_attn_key_values,
        }

    def prepare_llm_inputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        cross_attention_mask,
        full_text_row_masked_out_mask,
        past_key_values,
        cache_position,
        cross_attention_key_values,
    ):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cross_attention_mask": cross_attention_mask,
            "full_text_row_masked_out_mask": full_text_row_masked_out_mask,
            "cache_position": cache_position,
        }

        if past_key_values is None:
            self.request.reset_state()
            self.next_beam_idx = np.arange(input_ids.shape[0], dtype=int)
            self._past_length = 0

        model_inputs.update(dict(zip(self.lm_cross_attn_inputs, cross_attention_key_values)))
        if "beam_idx" in self.input_names:
            model_inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(input_ids.shape[0], dtype=int)

        return model_inputs

    def prepare_remote_tensors(self):
        context = core.get_default_context("GPU")
        for idx, name in enumerate(self.lm_cross_attn_inputs):
            remote_tensor = context.create_tensor(ov.Type.f16, ov.Shape([1, 32, 6404, 128]), {})
            self.vision_request.set_tensor(self.cross_attn_outputs[idx], remote_tensor)
            self.request.set_tensor(name, remote_tensor)


if __name__ == "__main__":
    model_id = "Llama-3.2-11B-Vision-Instruct/OV"
    LANGUAGE_MODEL_NAME = "llm_int4_asym_r10_gs64_max_activation_variance_all_layers.xml"
    IMAGE_ENCODER_NAME = "openvino_vision_encoder.xml"
    ov_model = OVMLlamaForConditionalGeneration(model_id, device="CPU", language_model_name=LANGUAGE_MODEL_NAME, image_encoder_name=IMAGE_ENCODER_NAME)
    processor = AutoProcessor.from_pretrained(model_id)
    url = "https://llava-vl.github.io/static/images/view.jpg"
    raw_image = Image.open(requests.get(url, stream=True).raw)
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe image in two sentences"}]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    url = "https://llava-vl.github.io/static/images/view.jpg"
    raw_image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=text, images=[raw_image], return_tensors="pt")
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    output = ov_model.generate(**inputs, do_sample=False, max_new_tokens=50, streamer=streamer)
    print(f"Visual encoder time {ov_model.vision_encoder_infer_time[0] * 1000} ms")
    print(f"First token latency {ov_model.llm_infer_time[0] * 1000}ms, Second token latency {np.mean(np.array(ov_model.llm_infer_time[1:])) * 1000}ms")
