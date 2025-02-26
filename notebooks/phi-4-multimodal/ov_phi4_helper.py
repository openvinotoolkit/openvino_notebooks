import openvino as ov
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationMixin, GenerationConfig, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPooling
import types
from pathlib import Path
import torch
from typing import List, Tuple, Optional, Union
import openvino.opset13 as opset13
import nncf
import numpy as np
from enum import Enum
import requests
from PIL import Image
import math
import shutil

VISION_EMBEDDINGS_PATH = "openvino_vision_embeddings_model.xml"
VISION_PROJECTOR_PATH = "openvino_vision_projection_model.xml"
TEXT_EMBEDDINGS_PATH = "openvino_text_embeddings_model.xml"
LM_PATH = "openvino_language_model.xml"
AUDIO_EMBEDDINGS_PATH = "openvino_audio_embeddings_model.xml"
AUDIO_FORWARD_EMBEDDINGS_PATH = "openvino_audio_forward_embeddings_model.xml"
AUDIO_ENCODER_PATH = "openvino_audio_encoder_model.xml"
AUDIO_VISION_PROJECTOR_PATH = "openvino_audio_vision_projection_model.xml"
AUDIO_SPEECH_PROJECTOR_PATH = "openvino_audio_text_projection_model.xml"

user_prompt = "<|user|>"
assistant_prompt = "<|assistant|>"
prompt_suffix = "<|end|>"
IMAGE_SPECIAL = "<|endoftext10|>"
AUDIO_SPECIAL = "<|endoftext11|>"


class InputMode(Enum):
    LANGUAGE = 0
    VISION = 1
    SPEECH = 2
    VISION_SPEECH = 3


_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>', or we can better name it (in `tokenizer_config.json`)
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999, -1]  # For backward compatibility
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE = [float("-inf"), -10000]  # For backward compatibility

quantization_config = {
    "vision": {"mode": nncf.CompressWeightsMode.INT8_ASYM},
    "llm": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 1.0},
    "audio": {"mode": nncf.CompressWeightsMode.INT8_ASYM},
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


def get_vision_position_ids(pixel_values, patch_attention_mask, patch_size=14, num_patches_per_side=32):
    batch_size = pixel_values.shape[0]
    max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
    max_nb_patches_h, max_nb_patches_w = max_im_h // patch_size, max_im_w // patch_size
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
    position_ids = torch.full(
        size=(
            batch_size,
            max_nb_patches_h * max_nb_patches_w,
        ),
        fill_value=0,
    )

    for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
        nb_patches_h = p_attn_mask[:, 0].sum()
        nb_patches_w = p_attn_mask[0].sum()

        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

        pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
        position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids
    return position_ids


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
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs[3:]]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs[1:]]
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


def unfold_tensor(xs_pad, max_seq_len):
    """
    For a given tensor with shape of (N, T, D), if sequence length T is longer than max_seq_len,
    this function unfold it to a (NT', max_seq_len, D) where T' is T // max_seq_len.
    Args:
        xs_pad: N, T, D
    """
    _, _, D = xs_pad.shape
    xs_pad = xs_pad.transpose(-1, -2)  # convert to N, D, T
    # N x D x 1 x T => N x (D x max_seq_len) x T'
    xs_pad = torch.nn.functional.unfold(
        xs_pad[..., None, :],
        kernel_size=(1, max_seq_len),
        stride=(1, max_seq_len),
    )

    new_bsz, _, slen = xs_pad.shape
    # N x D x max_seq_len x T'
    xs_pad = xs_pad.view(new_bsz, -1, max_seq_len, slen)
    # N x T' x max_seq_len x D
    xs_pad = xs_pad.permute(0, 3, 2, 1).contiguous()
    # NT' x max_seq_len x D
    xs_pad = xs_pad.view(-1, max_seq_len, D)
    return xs_pad


def convert_phi4o(input_dir, output_dir, quantization_config=None):
    output_dir = Path(output_dir)
    all_converted = all(
        [
            (output_dir / submodel_path).exists()
            for submodel_path in [
                VISION_EMBEDDINGS_PATH,
                VISION_PROJECTOR_PATH,
                TEXT_EMBEDDINGS_PATH,
                LM_PATH,
                AUDIO_EMBEDDINGS_PATH,
                AUDIO_FORWARD_EMBEDDINGS_PATH,
                AUDIO_ENCODER_PATH,
                AUDIO_SPEECH_PROJECTOR_PATH,
                AUDIO_VISION_PROJECTOR_PATH,
                AUDIO_ENCODER_PATH,
                AUDIO_FORWARD_EMBEDDINGS_PATH,
            ]
        ]
    )
    if all_converted:
        print(f"✅ {input_dir} model already converted. You can find results in {output_dir}")
        return
    print(f"⌛ {input_dir} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")
    config = AutoConfig.from_pretrained(input_dir, trust_remote_code=True)
    if "activation_checkpointing" in config.audio_processor["config"]:
        config.audio_processor["config"]["activation_checkpointing"] = ""
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(input_dir, config=config, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(input_dir, trust_remote_code=True)
    processor.chat_template = processor.tokenizer.chat_template
    model.generation_config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    if Path(input_dir).is_dir():
        shutil.copy(Path(input_dir) / "preprocessor_config.json", output_dir / "preprocessor_config.json")

    model.config.glb_GN = model.model.embed_tokens_extend.image_embed.glb_GN.tolist()
    model.config.sub_GN = model.model.embed_tokens_extend.image_embed.sub_GN.tolist()
    model.config.num_img_tokens = model.model.embed_tokens_extend.image_embed.num_img_tokens
    model.config.base_vision_feat_height_target = model.model.embed_tokens_extend.image_embed.base_feat_height_target
    model.config.base_vision_feat_height_reduction = model.model.embed_tokens_extend.image_embed.base_feat_height_reduction
    model.config.crop_size = model.model.embed_tokens_extend.image_embed.crop_size
    model.config.image_dim_out = model.model.embed_tokens_extend.image_embed.image_dim_out
    model.config.hd_transform_order = model.model.embed_tokens_extend.image_embed.hd_transform_order
    model.config.save_pretrained(output_dir)
    print("✅ Original model successfully loaded")

    if not (output_dir / TEXT_EMBEDDINGS_PATH).exists():
        print("⌛ Convert Input embedding model")
        ov_model = ov.convert_model(model.model.embed_tokens, example_input=torch.ones([2, 2], dtype=torch.long))
        ov.save_model(ov_model, output_dir / TEXT_EMBEDDINGS_PATH)
        print("✅ Input embedding model successfully converted")

    if not (output_dir / LM_PATH).exists():
        print("⌛ Convert Language model")

        def lm_forward(self, inputs_embeds, attention_mask, position_ids, past_key_values):
            num_logits_to_keep = 1
            from transformers.cache_utils import DynamicCache

            pkv = DynamicCache.from_legacy_cache(past_key_values)
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, use_cache=True, past_key_values=pkv)
            hidden_states = outputs[0]
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
            return (logits, outputs.past_key_values.to_legacy_cache())

        model.forward = types.MethodType(lm_forward, model)
        inputs_embeds = torch.zeros([2, 2, model.config.hidden_size], dtype=torch.float32)
        attention_mask = torch.ones([2, 4], dtype=torch.long)
        position_ids = torch.tensor([[2, 3], [2, 3]])
        pkv_input_names = []
        pkv_inputs = []
        pkv_output_names = []
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers
        shape = (2, model.config.num_key_value_heads, 2, head_dim)
        for idx in range(num_layers):
            pkv_inputs.append((torch.randn(shape), torch.randn(shape)))
            pkv_input_names.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
            pkv_output_names.extend([f"present.{idx}.key", f"present.{idx}.value"])

        model_inputs = ["inputs_embeds", "attention_mask", "position_ids", *pkv_input_names]
        model_outputs = ["logits", *pkv_output_names]

        dummy_inputs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "position_ids": position_ids, "past_key_values": pkv_inputs}
        ov_model = ov.convert_model(model, example_input=dummy_inputs)
        for input, input_name in zip(ov_model.inputs, model_inputs):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, model_outputs):
            output.get_tensor().set_names({output_name})

        patch_stateful(ov_model)
        if quantization_config is not None and "llm" in quantization_config:
            ov_model = nncf.compress_weights(ov_model, **quantization_config["llm"])
        ov.save_model(ov_model, output_dir / LM_PATH)
        print("✅ Language model successfully converted")

    print("⌛ Convert Audio embedding model")
    if not (output_dir / AUDIO_EMBEDDINGS_PATH).exists():
        ov_model = ov.convert_model(model.model.embed_tokens_extend.audio_embed.encoder.encoder_embedding, example_input=torch.ones([1, 1233, 80]))
        ov.save_model(ov_model, output_dir / AUDIO_EMBEDDINGS_PATH)
    if not (output_dir / AUDIO_FORWARD_EMBEDDINGS_PATH).exists():

        def forward(self, input_tensor):
            input_tensor, masks = self._forward_embeddings_core(input_tensor, None)
            return input_tensor

        model.model.embed_tokens_extend.audio_embed.encoder.forward = types.MethodType(forward, model.model.embed_tokens_extend.audio_embed.encoder)

        ov_model = ov.convert_model(model.model.embed_tokens_extend.audio_embed.encoder, example_input=torch.ones([1, 498, 80]))
        ov.save_model(ov_model, output_dir / AUDIO_FORWARD_EMBEDDINGS_PATH)

    if not (output_dir / AUDIO_ENCODER_PATH).exists():

        def forward(self, input_tensor, hs_mask):
            relative_attention_bias = self.init_relative_attention_bias(input_tensor)

            _simplified_path = self.extra_layer_output_idx == -1 and relative_attention_bias is None

            if _simplified_path:
                input_tensor, *_ = self.encoders(input_tensor, None, None, hs_mask)
            else:
                for i, layer in enumerate(self.encoders):
                    input_tensor, _, _, _ = layer(
                        input_tensor,
                        None,
                        None,
                        hs_mask,
                        relative_attention_bias=relative_attention_bias,
                    )

                    if i == self.extra_layer_output_idx:
                        layer_emb = input_tensor
            return input_tensor

        model.model.embed_tokens_extend.audio_embed.encoder.forward = types.MethodType(forward, model.model.embed_tokens_extend.audio_embed.encoder)

        ov_model = ov.convert_model(
            model.model.embed_tokens_extend.audio_embed.encoder, example_input=(torch.ones([1, 63, 1024]), torch.ones([1, 63, 63], dtype=torch.bool))
        )

        if quantization_config and "audio" in quantization_config:
            ov_model = nncf.compress_weights(ov_model, **quantization_config["audio"])
        ov.save_model(ov_model, output_dir / AUDIO_ENCODER_PATH)
    if not (output_dir / AUDIO_SPEECH_PROJECTOR_PATH).exists():
        ov_model = ov.convert_model(model.model.embed_tokens_extend.audio_embed.audio_projection["speech"], example_input=torch.ones([1, 155, 1024]))
        ov.save_model(ov_model, output_dir / AUDIO_SPEECH_PROJECTOR_PATH)
    if not (output_dir / AUDIO_VISION_PROJECTOR_PATH).exists():
        ov_model = ov.convert_model(model.model.embed_tokens_extend.audio_embed.audio_projection["vision"], example_input=torch.ones([1, 155, 1024]))
        ov.save_model(ov_model, output_dir / AUDIO_VISION_PROJECTOR_PATH)

    print("✅ Audio embedding model successfully converted")
    print("⌛ Convert Image embedding model")
    vision_embed_model = model.model.embed_tokens_extend.image_embed
    if not (output_dir / VISION_EMBEDDINGS_PATH).exists():
        prompt = f"{user_prompt}{IMAGE_SPECIAL}What is shown in this image?{prompt_suffix}{assistant_prompt}"
        image_path = Path("cat.png")

        if not image_path.exists():
            url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
            image = Image.open(requests.get(url, stream=True).raw)
            image.save(image_path)
        else:
            image = Image.open(image_path)
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        pixel_values = inputs["input_image_embeds"].flatten(0, 1)
        attention_mask = inputs["image_attention_mask"].flatten(0, 1).to(torch.bool)
        position_ids = get_vision_position_ids(pixel_values, attention_mask)

        def get_img_features(self, img_embeds: torch.FloatTensor, attention_mask=None, position_ids=None) -> torch.FloatTensor:
            LAYER_IDX = self.layer_idx
            TYPE_FEATURE = self.type_feature

            if self.freeze_img_processor:
                with torch.no_grad():
                    if attention_mask is not None:
                        img_processor_output = self.img_processor(
                            img_embeds, output_hidden_states=True, patch_attention_mask=attention_mask, position_ids=position_ids
                        )
                    else:
                        img_processor_output = self.img_processor(img_embeds, output_hidden_states=True, position_ids=position_ids)
                    img_feature = img_processor_output.hidden_states[LAYER_IDX]
            else:
                if attention_mask is not None:
                    img_processor_output = self.img_processor(
                        img_embeds, output_hidden_states=True, patch_attention_mask=attention_mask, position_ids=position_ids
                    )
                else:
                    img_processor_output = self.img_processor(img_embeds, output_hidden_states=True, position_ids=position_ids)
                img_feature = img_processor_output.hidden_states[LAYER_IDX]

            if TYPE_FEATURE == "patch":
                patch_feature = img_feature
                if self.image_token_compression is not None:
                    # reshape to 2D tensor
                    width = int(math.sqrt(patch_feature.size(1)))
                    patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
                    # convert to NCHW
                    patch_feature = patch_feature.permute(0, 3, 1, 2)
                    if getattr(self, "img_processor_padding", None) is not None:
                        patch_feature = self.img_processor_padding(patch_feature)
                    patch_feature = self.image_token_compression(patch_feature)
                    # convert to NHWC
                    patch_feature = patch_feature.permute(0, 2, 3, 1)
                    patch_feature = patch_feature.view(-1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1))
                elif getattr(self, "img_processor_padding", None) is not None:
                    width = int(math.sqrt(patch_feature.size(1)))
                    patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
                    # convert to NCHW
                    patch_feature = patch_feature.permute(0, 3, 1, 2)
                    patch_feature = self.img_processor_padding(patch_feature)
                    # convert to NHWC
                    patch_feature = patch_feature.permute(0, 2, 3, 1)
                    patch_feature = patch_feature.view(-1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1))
                return patch_feature

            if TYPE_FEATURE == "cls_patch":
                if self.image_token_compression is not None:
                    # reshape to 2D tensor
                    patch_feature = img_feature[:, 1:]
                    cls_feature = img_feature[:, 0]
                    width = math.sqrt(patch_feature.size(1))
                    patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
                    patch_feature = self.image_token_compression(patch_feature)
                    patch_feature = patch_feature.view(-1, patch_feature.size(-2) * patch_feature.size(-1))
                    img_feature = torch.cat([cls_feature, patch_feature], dim=1)
                return img_feature

        vision_embed_model.forward = types.MethodType(get_img_features, vision_embed_model)

        def transformer_fwd(
            self,
            pixel_values,
            patch_attention_mask: Optional[torch.BoolTensor] = None,
            position_ids: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
            from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            batch_size = pixel_values.size(0)
            if patch_attention_mask is None:
                patch_attention_mask = torch.ones(
                    size=(
                        batch_size,
                        pixel_values.size(2) // self.config.patch_size,
                        pixel_values.size(3) // self.config.patch_size,
                    ),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )

            hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask, position_ids=position_ids)

            patch_attention_mask = patch_attention_mask.view(batch_size, -1)
            # The call to `_upad_input` in `_flash_attention_forward` is expensive
            # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
            # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
            if not torch.any(~patch_attention_mask):
                attention_mask = None
            else:
                attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.post_layernorm(last_hidden_state)

            pooled_output = self.head(
                hidden_state=last_hidden_state,
                attention_mask=patch_attention_mask,
            )

            if not return_dict:
                return (last_hidden_state, pooled_output) + encoder_outputs[1:]

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        def attn_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

            if output_attentions:
                return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            batch_size, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and attention_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

            attn_output = self.out_proj(attn_output)

            return attn_output, None

        def embd_forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor, position_ids: torch.FloatTensor = None) -> torch.Tensor:
            batch_size = pixel_values.size(0)

            patch_embeds = self.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            if position_ids is None:
                max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
                max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
                boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
                position_ids = torch.full(
                    size=(
                        batch_size,
                        max_nb_patches_h * max_nb_patches_w,
                    ),
                    fill_value=0,
                )

                for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
                    nb_patches_h = p_attn_mask[:, 0].sum()
                    nb_patches_w = p_attn_mask[0].sum()

                    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
                    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

                    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                    pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
                    position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

            position_ids = position_ids.to(self.position_embedding.weight.device)

            embeddings = embeddings + self.position_embedding(position_ids)
            return embeddings

        for layer in vision_embed_model.img_processor.encoder.layers:
            layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)
        vision_embed_model.img_processor.forward = types.MethodType(transformer_fwd, vision_embed_model.img_processor)
        vision_embed_model.img_processor.embeddings.forward = types.MethodType(embd_forward, vision_embed_model.img_processor.embeddings)

        ov_model = ov.convert_model(
            vision_embed_model, example_input={"img_embeds": pixel_values, "attention_mask": attention_mask, "position_ids": position_ids}
        )
        if quantization_config and "vision" in quantization_config:
            ov_model = nncf.compress_weights(ov_model, **quantization_config["vision"])
        ov.save_model(ov_model, output_dir / VISION_EMBEDDINGS_PATH)

    if not (output_dir / VISION_PROJECTOR_PATH).exists():
        ov_model = ov.convert_model(vision_embed_model.img_projection, example_input=torch.zeros([1, 1841, 1152]))
        ov.save_model(ov_model, output_dir / VISION_PROJECTOR_PATH)

    print("✅ Image embedding model successfully converted")
    print(f"✅ Model successfully converted and can be found in {output_dir}")


def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
    """
    The function is very important for Transformer Transducer Streaming mode
    Args:
        xs_len (int): sequence length
        chunk_start_idx (list): first idx of each chunk, such as [0,18,36,48]. It also supports adaptive chunk size [0,10,15,45]
        left_window (int): how many left chunks can be seen
        right_window (int): how many right chunks can be seen. It is used for chunk overlap model.
        Returns:
            mask (torch.Tensor): a mask tensor for streaming model
            Torch 1.0.1
            tensor([[1., 1., 0., 0.],
                    [0., 1., 1., 0.],
                    [0., 0., 1., 1.]])
            Torch 1.4.1
            tensor([[True., True., False., False.],
                    [False., True., True., False.],
                    [False., False., True., True.]])
    """
    chunk_start_idx = torch.Tensor(chunk_start_idx).long()  # first idx of each chunk, such as [0,18,36,48].
    start_pad = torch.nn.functional.pad(chunk_start_idx, (1, 0))  # append 0 to the beginning, so it becomes [0, 0, 18, 36, 48]
    end_pad = torch.nn.functional.pad(chunk_start_idx, (0, 1), value=x_len)  # append x_len to the end, so it becomes [0,18,36,48, x_len]
    seq_range = torch.arange(0, x_len).unsqueeze(-1)  # seq_range size: [x_len, 1]
    idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]  # idx size: [x_len]
    boundary = end_pad[idx]  # boundary size: [x_len]
    seq_range_expand = torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)  # seq_range_expand size [x_len, x_len]
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    boundary_left = start_pad[idx_left]
    mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
    boundary_right = end_pad[idx_right]
    mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
    return mask_left & mask_right


core = ov.Core()


class OVModelForCausalLMWithEmb(GenerationMixin):
    def __init__(self, model_dir, device="CPU", config=None, ov_config=None, compile=True) -> None:
        self._supports_cache_class = False
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True) if config is None else config
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False
        self.generation_config = GenerationConfig.from_model_config(self.config)
        model_dir = Path(model_dir)
        self.model = core.read_model(model_dir / LM_PATH)
        self.token_emb = core.read_model(model_dir / TEXT_EMBEDDINGS_PATH)
        self.request = None
        self.token_emb_request = None
        self._device = device.upper()
        self.device = torch.device("cpu")
        self.ov_config = ov_config
        self.next_beam_idx = None
        self._past_length = None
        self.input_names = [input_t.get_any_name() for input_t in self.model.inputs]
        self.main_input_name = "input_ids"
        if compile:
            self.compile()

    def compile(self):
        if self.request is None:
            self.request = core.compile_model(self.model, self._device, self.ov_config).create_infer_request()
        self._compile_token_emb()

    def _compile_token_emb(self):
        if self.token_emb_request is None:
            self.token_emb_request = core.compile_model(self.token_emb, self._device, self.ov_config)

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
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        inputs = {}
        # past_key_values are not used explicitly, instead they are handled inside the model
        if past_key_values is None:
            # This is the first iteration in a sequence, reset all states
            if self.request is not None:
                self.request.reset_state()
                # Set initial value for the next beam_idx input that will be used at the current iteration
                # and will be optionally updated by _reorder_cache at the next iterations if beam_search is used
                self.next_beam_idx = np.arange(batch_size, dtype=int)
                self._past_length = 0
        past_len = self._get_past_length(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids if past_key_values is None else input_ids[:, -1:])

            if hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb
        inputs["inputs_embeds"] = inputs_embeds

        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names or "position_ids" in self.input_names:
            if attention_mask is not None:
                attention_mask = np.array(attention_mask)
            else:
                attention_mask = np.ones((inputs_embeds.shape[0], inputs_embeds.shape[1] + past_len), dtype=int)

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids = np.array(position_ids)
            else:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

            inputs["position_ids"] = position_ids

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

        return inputs

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
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
        logits = self.request.get_tensor("logits").data
        logits = torch.from_numpy(logits).to(self.device)
        past_key_values = ((),)
        self._past_length += inputs["inputs_embeds"].shape[1]

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    # Adapted from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)
        past_len = 0

        if past_key_values is not None:
            past_len = self._get_past_length(past_key_values)
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and input_ids is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_len) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif input_ids is not None and past_len < input_ids.shape[1]:
                input_ids = input_ids[:, past_len:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None and "position_ids" in self.input_names:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values and input_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        cache_position = torch.arange(past_len, past_len + position_ids.shape[-1], device=position_ids.device)

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds if past_key_values is None else None,
            "cache_position": cache_position,
        }

        return model_inputs

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        return self._past_length

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""

        return True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class OVPhioModelForCausalLM(GenerationMixin):
    def __init__(self, model_dir, device="CPU", ov_config=None) -> types.NoneType:
        self._supports_cache_class = False
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.model = OVModelForCausalLMWithEmb(model_dir, device, ov_config)
        self.vision_embedings = core.compile_model(model_dir / VISION_EMBEDDINGS_PATH, device, ov_config)
        self.vision_projector = core.compile_model(model_dir / VISION_PROJECTOR_PATH, device, ov_config)
        self.audio_embeddings = core.compile_model(model_dir / AUDIO_EMBEDDINGS_PATH, device, ov_config)
        self.audio_forward_embeddings = core.compile_model(model_dir / AUDIO_FORWARD_EMBEDDINGS_PATH, device, ov_config)
        self.audio_encoder = core.compile_model(model_dir / AUDIO_ENCODER_PATH, device, ov_config)
        self.audio_vision_projector = core.compile_model(model_dir / AUDIO_VISION_PROJECTOR_PATH, device, ov_config)
        self.audio_speech_projector = core.compile_model(model_dir / AUDIO_SPEECH_PROJECTOR_PATH)
        self.sub_GN = torch.tensor(self.config.sub_GN)
        self.glb_GN = torch.tensor(self.config.glb_GN)
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self._device = device
        self.chunk_size = -1
        self.left_chunk = 18
        self.time_reduction = 8

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""

        return True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        return self.model._reorder_cache(past_key_values, beam_idx)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask=None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        input_mode=None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        audio_projection_mode = None
        if input_audio_embeds is not None:
            if isinstance(input_mode, torch.Tensor):
                assert len(input_mode) == 1
                input_mode = input_mode[0].item()
            input_mode = InputMode(input_mode)

            if input_mode in [InputMode.VISION_SPEECH, InputMode.VISION]:
                audio_projection_mode = "vision"
            elif input_mode == InputMode.SPEECH:
                audio_projection_mode = "speech"
            elif input_mode == InputMode.LANGUAGE:
                audio_projection_mode = "speech"
            else:
                raise ValueError(f"Invalid input_mode: {input_mode}")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens_extend(
                input_ids=input_ids,
                input_embeds=inputs_embeds,
                input_image_embeds=input_image_embeds,
                input_audio_embeds=input_audio_embeds,
                image_sizes=image_sizes,
                image_attention_mask=image_attention_mask,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                audio_projection_mode=audio_projection_mode,
                past_key_values=past_key_values,
            )
        return self.model.forward(
            input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values
        )

    def embed_tokens_extend(
        self,
        input_ids: torch.LongTensor,
        input_embeds,
        input_image_embeds: torch.FloatTensor = None,
        input_audio_embeds: torch.FloatTensor = None,
        image_sizes=None,
        image_attention_mask=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
        past_key_values=None,
    ):
        if past_key_values is not None:
            return self.model.embed_tokens(input_ids)

        MAX_INPUT_ID = int(1e9)

        new_input_ids = input_ids.clone()
        new_input_ids[(input_ids >= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[0]) & (input_ids <= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[1])] = (
            _IMAGE_SPECIAL_TOKEN_ID
        )
        new_input_ids[(input_ids >= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[0]) & (input_ids <= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[1])] = (
            _AUDIO_SPECIAL_TOKEN_ID
        )
        input_ids = new_input_ids
        image_position_mask = input_ids == _IMAGE_SPECIAL_TOKEN_ID
        non_image_position_mask = ~image_position_mask
        image_hidden_states = self.image_embed(
            input_ids=input_ids, input_embeds=input_image_embeds, image_sizes=image_sizes, image_attention_mask=image_attention_mask
        )
        audio_hidden_states = self.audio_embed(
            input_ids=input_ids,
            input_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            audio_projection_mode=audio_projection_mode,
        )
        hidden_states = image_hidden_states * image_position_mask.unsqueeze(-1) + audio_hidden_states * non_image_position_mask.unsqueeze(-1)

        return hidden_states

    def image_embed(self, input_ids: torch.LongTensor, input_embeds: torch.FloatTensor, image_sizes=None, **kwargs):
        if isinstance(input_ids, tuple):
            # # pipeline parallel
            input_ids, input_embeds = input_ids

        img_embeds = input_embeds
        if image_sizes is None and "image_sizes" in kwargs:
            image_sizes = kwargs["image_sizes"]
        img_sizes = image_sizes
        if "image_attention_mask" in kwargs:
            image_attention_mask = kwargs["image_attention_mask"]
        else:
            image_attention_mask = None
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        with torch.no_grad():
            positions = torch.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=False)
            positions_tuple = torch.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=True)

        # logger.info(f'position size: {positions.size()} ...')
        select = False
        hd_transform = False
        if len(positions.tolist()) > 0:
            if self.config.embd_layer["image_embd_layer"]["use_hd_transform"] and img_sizes is not None and len(img_sizes):
                hd_transform = True
                assert img_embeds.ndim == 5, f"(branch 1) img_embeds size: {img_embeds.size()}, expect 5D tensor for hd transform"
                # img_embeds: (num_images, max_num_crops, 3, H, W)
                # img_sizes: (num_images, 2).view(1, -1)

                bs = img_embeds.shape[0]
                pixel_values = img_embeds.flatten(0, 1)
                patch_attn_mask = image_attention_mask.type(torch.BoolTensor).flatten(0, 1)
                v_position_ids = get_vision_position_ids(pixel_values, patch_attn_mask)
                # Nx(HW)xC
                img_features = torch.from_numpy(self.vision_embedings([pixel_values, patch_attn_mask, v_position_ids])[0])

                base_feat_height_target = self.config.base_vision_feat_height_target
                base_resolution = self.config.crop_size
                base_feat_height_reduction = self.config.base_vision_feat_height_reduction

                base_feat_height = base_feat_width = int(np.sqrt(img_features.shape[1]))

                assert (
                    base_feat_height == base_feat_height_target and base_feat_width == base_feat_height_target
                ), f"base_feat_height: {base_feat_height}, base_feat_width: {base_feat_width}, expect {base_feat_height_target} features for hd transform"

                # bs x max_num_crops x (24x24) x C
                img_features = img_features.view(bs, -1, base_feat_height * base_feat_width, self.config.image_dim_out)
                C = self.config.image_dim_out
                H = base_feat_height

                output_imgs = []
                output_len = []
                # training is tensor, inference is list
                if isinstance(img_sizes, torch.Tensor):
                    img_sizes = img_sizes.view(-1, 2)
                for _bs in range(bs):
                    h, w = img_sizes[_bs]
                    h = h // base_resolution
                    w = w // base_resolution
                    B_ = h * w

                    # 1 x (24x24) x 1024
                    global_img_feature = img_features[_bs, :1]

                    # 1 x 12 x 12 x 4096
                    glb_img = (
                        global_img_feature.reshape(1, H, H, C)
                        .reshape(1, H // base_feat_height_reduction, base_feat_height_reduction, H // base_feat_height_reduction, base_feat_height_reduction, C)
                        .contiguous()
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(
                            1, H // base_feat_height_reduction, H // base_feat_height_reduction, base_feat_height_reduction * base_feat_height_reduction * C
                        )
                        .contiguous()
                    )
                    temp_glb_GN = self.sub_GN.repeat(1, H // base_feat_height_reduction, 1, 1)

                    # 1 x 156 x 4096
                    glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1, -1, base_feat_height_reduction * base_feat_height_reduction * C)

                    # (max_num_crops-1) x (12x12) x C
                    sub_img = img_features[_bs, 1:]
                    # 16x574x1024
                    # get rid of padding sub_img
                    sub_img = sub_img[:B_]

                    # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
                    sub_img = (
                        sub_img.reshape(B_, H, H, C)
                        .reshape(
                            B_, H // base_feat_height_reduction, base_feat_height_reduction, H // base_feat_height_reduction, base_feat_height_reduction, C
                        )
                        .contiguous()
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(B_, -1, base_feat_height_reduction * base_feat_height_reduction * C)
                        .contiguous()
                    )
                    sub_img = (
                        sub_img.reshape(1, h, w, base_feat_height // base_feat_height_reduction, base_feat_width // base_feat_height_reduction, -1)
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(
                            1,
                            h * base_feat_height // base_feat_height_reduction,
                            w * base_feat_width // base_feat_height_reduction,
                            base_feat_height_reduction * base_feat_height_reduction * C,
                        )
                    )

                    if image_attention_mask is not None and len(image_attention_mask) > 0:
                        reshaped_image_attention_mask = (
                            image_attention_mask[_bs, 1 : B_ + 1, 0::2, 0::2]
                            .reshape(1, h, w, base_feat_height // base_feat_height_reduction, base_feat_width // base_feat_height_reduction)
                            .permute(0, 1, 3, 2, 4)
                            .reshape(1, h * base_feat_height // base_feat_height_reduction, w * base_feat_width // base_feat_height_reduction)
                        )
                        useful_height = int(reshaped_image_attention_mask[0, :, 0].sum().item())
                        useful_width = int(reshaped_image_attention_mask[0, 0, :].sum().item())
                        sub_img = sub_img[:, :useful_height, :useful_width]
                        temp_sub_GN = self.sub_GN.repeat(1, useful_height, 1, 1)
                        temp_len = (
                            int(image_attention_mask[_bs, : B_ + 1, 0::2, 0::2].sum().item())
                            + (useful_height + 1)
                            + base_feat_height // base_feat_height_reduction
                        )
                    else:
                        temp_sub_GN = self.sub_GN.repeat(1, h * base_feat_height // base_feat_height_reduction, 1, 1)
                        temp_len = int((h * w + 1) * self.num_img_tokens + 1 + (h + 1) * base_feat_height // base_feat_height_reduction)

                    sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1, -1, base_feat_height_reduction * base_feat_height_reduction * C)
                    # (1, num_img_tokens, 1024*4)

                    # glb + sub
                    if self.config.hd_transform_order == "glb_sub":
                        output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                    elif self.config.hd_transform_order == "sub_glb":
                        output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
                    else:
                        raise NotImplementedError(f"hd_transform_order = {self.hd_transform_order}, not implemented")

                    # temp_len = int((h*w+1)*144 + 1 + (h+1)*12)
                    assert temp_len == output_imgs[-1].shape[1], f"temp_len: {temp_len}, output_imgs[-1].shape[1]: {output_imgs[-1].shape[1]}"
                    output_len.append(temp_len)

                img_set_tensor = torch.from_numpy(self.vision_projector(output_imgs)[0])
                # for _output_img in output_imgs:
                #     img_feature_proj = torch.from_numpy(self.vision_projector(_output_img)[0])
                #     img_set_tensor.append(img_feature_proj)

            else:
                raise NotImplementedError
            select = True

        # we use the token embedding layer from the huggingface model, this is REQUIRED to make sure we are using the loaded weights.
        hidden_states = torch.from_numpy(self.model.embed_tokens(input_ids))

        if select:
            if hd_transform:
                # new implementation without in-place operation
                # Ref: https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/4a0d683eba9f1d0cbfb6151705d1ee73c25a80ca/modeling_phi3_v.py#L233
                # Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.index_put.html
                # Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.index_put_.html#torch.Tensor.index_put_
                # img_set_tensor: a list of tensors, each tensor has shape (1, N_tokens, C)
                # assert all([_img_set_tensor.shape[0] == 1 for _img_set_tensor in img_set_tensor]), 'img_set_tensor should have shape (1, N_tokens, C)'
                # Shape: (merged_N_tokens, C)
                merged_img_set_tensor = img_set_tensor.squeeze(0)  # torch.cat(img_set_tensor, dim=1).squeeze(0)
                merged_img_set_tensor = merged_img_set_tensor.to(hidden_states.dtype).to(hidden_states.device)
                # Temporarily disable autocast to avoid issue on bf16 tensors
                # Ref: https://github.com/pytorch/pytorch/issues/132715
                new_hidden_states = hidden_states.index_put(indices=positions_tuple, values=merged_img_set_tensor, accumulate=False)
                hidden_states = new_hidden_states
            else:
                raise NotImplementedError

        return hidden_states

    def audio_embed(
        self,
        input_ids: torch.LongTensor,
        input_embeds: torch.FloatTensor,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
        **kwargs,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        positions = torch.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID, as_tuple=False)
        positions_tuple = torch.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID, as_tuple=True)
        if len(positions.tolist()) > 0:
            audio_set_tensor = self.get_audio_features(input_embeds, audio_attention_mask, audio_projection_mode)

        hidden_states = torch.from_numpy(self.model.embed_tokens(input_ids))

        if len(positions.tolist()) > 0:

            assert audio_embed_sizes.sum().item() == len(
                positions
            ), f"please ensure the encoder outputs have the same length as defined in input_ids! \n audio_embed_sizes.sum().item(): {audio_embed_sizes.sum().item()} \n len(positions): {len(positions)} \n audio_embed_sizes: {audio_embed_sizes} \n positions: {positions} \n input_ids.shape \n {input_ids.shape}"

            # new implementation without in-place operation
            # Ref: https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/4a0d683eba9f1d0cbfb6151705d1ee73c25a80ca/modeling_phi3_v.py#L233
            # Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.index_put.html
            # Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.index_put_.html#torch.Tensor.index_put_
            # audio_set_tensor: shape (N_audios, N_padded_tokens, C)
            # Shape: (merged_N_tokens, C)
            merged_audio_set_tensor = torch.cat([audio_set_tensor[i, : audio_embed_sizes[i], :] for i in range(len(audio_embed_sizes))], dim=0)
            new_hidden_states = hidden_states.index_put(indices=positions_tuple, values=merged_audio_set_tensor, accumulate=False)
            hidden_states = new_hidden_states

        return hidden_states

    def get_audio_features(self, input_embeds: torch.FloatTensor, audio_attention_mask: torch.Tensor, audio_projection_mode: str = "speech"):
        xs_pad = self.audio_embeddings(input_embeds)[0]
        input_tensor, pos_k, pos_v, hs_mask, masks = self.forward_embeddings(xs_pad)

        unfolded = False
        ori_bz, seq_len, D = input_tensor.shape
        max_seq_len = 500  # maxium position for absolute positional encoding
        masks_unfold = None
        if seq_len > max_seq_len:
            # audio sequence is longer than max_seq_len, unfold it into chunks of max_seq_len
            unfolded = True
            # the unfold op will drop residual frames, pad it to the multiple of max_seq_len
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0
            if chunk_pad_size > 0:
                input_tensor_pad = torch.nn.functional.pad(torch.from_numpy(input_tensor), (0, 0, 0, chunk_pad_size), "constant", 0)
                input_tensor = input_tensor_pad

            input_tensor = unfold_tensor(input_tensor, max_seq_len)
            if masks is not None:
                # revise hs_mask here because the previous calculated hs_mask did not consider extra pad
                subsampled_pad_mask = masks.squeeze(1)  # [bz, subsampled_unmask_seq_len]
                extra_padded_subsamlped_pad_mask = torch.nn.functional.pad(
                    subsampled_pad_mask, (0, chunk_pad_size), "constant", False
                )  # extra padding to the pad mask
                extra_padded_subsamlped_pad_mask = extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()
                masks_unfold = unfold_tensor(extra_padded_subsamlped_pad_mask, max_seq_len)  # unfold the pad mask like we did to the input tensor
                masks_unfold = masks_unfold.squeeze(-1).bool()  # unfold op does not support bool tensor
            else:
                masks_unfold = None
        hs_mask = self.calculate_hs_mask(input_tensor, masks_unfold)
        audio_features = self.audio_encoder([input_tensor, hs_mask])[0]
        if unfolded:
            embed_dim = audio_features.shape[-1]
            audio_features = np.reshape(audio_features, (ori_bz, -1, embed_dim))
            # if we ever padded before unfolding, we need to remove the padding
            if chunk_pad_size > 0:
                audio_features = audio_features[:, :-chunk_pad_size, :]
        audio_encoder = self.audio_vision_projector if audio_projection_mode == "vision" else self.audio_speech_projector
        audio_set_tensor = audio_encoder(audio_features)[0]

        return torch.from_numpy(audio_set_tensor)

    def _chunk_size_selection(self, chunk_size=None, left_chunk=None):
        """If chunk size is a list, we will randomly select a chunk size."""
        if isinstance(chunk_size, list):
            # Variable chunk size during training
            chunk_size_index = int(torch.randint(low=0, high=len(chunk_size), size=(1,)))
            chunk_size_train_eff = chunk_size[chunk_size_index]
            if not isinstance(left_chunk, list):
                raise ValueError("Since chunk_size is a list, left_chunk must be a list")
            if len(left_chunk) != len(chunk_size):
                raise ValueError("The length of left_chunk must be the same as length of chunk_size.")
            left_chunk_train_eff = left_chunk[chunk_size_index]
        else:
            chunk_size_train_eff = chunk_size
            left_chunk_train_eff = left_chunk

        return chunk_size_train_eff, left_chunk_train_eff

    def forward_embeddings(self, xs_pad, masks=None, chunk_size_nc=None, left_chunk_nc=None):
        """Forwarding the inputs through the top embedding layers

        Args:
            xs_pad: torch.Tensor
                input tensor
            masks: torch.Tensor
                input mask
            chunk_size_nc: (optional, default is None) chunk size for non-causal layers
            left_chunk_nc: (optional, default is None) # of left chunks for non-causal layers
        """
        # pylint: disable=R0915
        # get new lens.
        seq_len = int(self.compute_lens_change(xs_pad.shape[1]))
        if seq_len <= 0:
            raise ValueError(
                f"""The squence length after time reduction is invalid: {seq_len}.
                Your input feature is too short. Consider filtering out the very
                short sentence from data loader""",
            )

        batch_size = xs_pad.shape[0]

        enc_streaming_mask = self._streaming_mask(seq_len, batch_size, self.chunk_size, self.left_chunk)

        input_tensor = xs_pad

        input_tensor = self.audio_forward_embeddings(input_tensor)[0]

        streaming_mask = enc_streaming_mask
        if streaming_mask is not None and masks is not None:
            hs_mask = masks & streaming_mask
        else:
            hs_mask = streaming_mask

        if chunk_size_nc is not None:
            enc_streaming_mask_nc = self._streaming_mask(seq_len, batch_size, chunk_size_nc, left_chunk_nc)
            if masks is not None:
                hs_mask_nc = masks & enc_streaming_mask_nc
            else:
                hs_mask_nc = enc_streaming_mask_nc
        else:
            hs_mask_nc = None

        if chunk_size_nc is None:
            return input_tensor, None, None, hs_mask, None
        return input_tensor, None, None, hs_mask, None, hs_mask_nc

    def _streaming_mask(self, seq_len, batch_size, chunk_size, left_chunk):
        chunk_size_train_eff, left_chunk_train_eff = self._chunk_size_selection(chunk_size, left_chunk)

        # Create mask matrix for streaming
        # S stores start index. if chunksize is 18, s is [0,18,36,....]
        chunk_start_idx = np.arange(0, seq_len, chunk_size_train_eff)
        # avoid randomness when run evaluation or decoding

        enc_streaming_mask = adaptive_enc_mask(seq_len, chunk_start_idx, left_window=left_chunk_train_eff).unsqueeze(0).expand([batch_size, -1, -1])
        return enc_streaming_mask

    def compute_lens_change(self, feature_lens):
        """feature_lens: int
        return updated feature lens.

        This used to return a different lambda function for each case that computed
        the right thing.  That does not work within Torchscript.  If you really
        need this to be faster, create nn.Module()-s for all the cases and return
        one of them.  Torchscript does support that.
        """
        if self.config.audio_processor["config"]["input_layer"] == "nemo_conv":
            nemo_conv_settings = self.config.audio_processor["config"]["nemo_conv_settings"]
            # Handle the special causal case
            subsampling_causal_cond = nemo_conv_settings.get("subsampling", "dw_striding") in [
                "dw_striding",
                "striding",
                "striding_conv1d",
            ]
            is_causal = nemo_conv_settings.get("is_causal", False)
            if is_causal and subsampling_causal_cond:
                lens_change = (
                    torch.ceil(feature_lens / self.time_reduction).long()
                    if isinstance(feature_lens, torch.Tensor)
                    else math.ceil(feature_lens / self.time_reduction)
                )
                feature_lens_remainder = feature_lens % self.time_reduction
                if isinstance(feature_lens, torch.Tensor):
                    lens_change[feature_lens_remainder != 1] += 1
                elif feature_lens_remainder != 1:
                    lens_change += 1
                return lens_change
            ceil_func = math.ceil if isinstance(feature_lens, int) else torch.ceil
            return ceil_func(feature_lens / self.time_reduction)

    def calculate_hs_mask(self, xs_pad, mask):
        max_audio_length = xs_pad.shape[1]
        batch_size = xs_pad.shape[0]
        enc_streaming_mask = self._streaming_mask(max_audio_length, batch_size, self.chunk_size, self.left_chunk)
        if mask is None:
            return enc_streaming_mask

        feature_lens = mask.sum(1)
        padding_length = feature_lens
        pad_mask = torch.arange(0, max_audio_length).expand(padding_length.size(0), -1) < padding_length.unsqueeze(1)
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask & enc_streaming_mask
        return pad_mask

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        input_image_embeds=None,
        image_sizes=None,
        image_attention_mask=None,
        input_audio_embeds=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        input_mode=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        if past_key_values is None:
            model_inputs.update(
                {
                    "input_image_embeds": input_image_embeds,
                    "image_attention_mask": image_attention_mask,
                    "image_sizes": image_sizes,
                    "input_audio_embeds": input_audio_embeds,
                    "audio_embed_sizes": audio_embed_sizes,
                    "input_mode": input_mode,
                }
            )

        return model_inputs
