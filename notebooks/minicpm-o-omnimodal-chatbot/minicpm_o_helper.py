import torch
from threading import Thread
from copy import deepcopy
import shutil
import json
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, TextIteratorStreamer
from transformers.generation import GenerationMixin
from transformers import AutoConfig, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from pathlib import Path
from huggingface_hub import snapshot_download
import types
from typing import Optional, Union

try:
    from openvino import opset13
except ImportError:
    from openvino.runtime import opset13
import openvino as ov
import numpy as np
import gc

try:
    from openvino.passes import Manager, MatcherPass, WrapType, Matcher
except ImportError:
    from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher
import time

text_emb_path = Path("language_model/embed_tokens.xml")
llm_path = Path("language_model/language_model.xml")
image_encoder_path = Path("image_encoder.xml")
resampler_path = Path("resampler.xml")
audio_encoder_path = Path("audio_encoder.xml")
proejector_path = Path("proejector.xml")


def copy_llm_files(src_dir, dst_dir):
    for f in src_dir.glob("*"):
        if f.name in ("language_model.xml", "language_model.bin"):
            continue
        shutil.copy(f, dst_dir / f.name)


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
                slice = opset13.slice(grand_parent, start, stop, step, axes, name="inserted_slice")
                for consumer in consumers:
                    consumer.replace_source_output(slice.output(0))
                self.model_changed = True
                # Use new operation for additional matching
                self.register_new_node(slice)
                print("applied slice for lm head")

                return True

        self.register_matcher(Matcher(param, "InsertSlice"), callback)


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
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})
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


def patch_stateful(ov_model):
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs[2:-1]]
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


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


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
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
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
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


def patch_model_code(orig_model_dir):
    model_file = orig_model_dir / "modeling_navit_siglip.py"
    orig_model_file = model_file.parent / ("orig_" + model_file.name)
    if not orig_model_file.exists():
        model_file.rename(orig_model_file)
        with orig_model_file.open("r") as f:
            content = f.read()
            content = content.replace("if is_flash_attn_2_available():", "")
            content = content.replace("from flash_attn import flash_attn_func", "")
            content = content.replace("from flash_attn.bert_padding import index_first_axis", "")
            content = content.replace("from flash_attn.bert_padding import pad_input", "")
            content = content.replace("from flash_attn.bert_padding import unpad_input", "")

            with model_file.open("w") as out_f:
                out_f.write(content)
    resampler_file = orig_model_dir / "resampler.py"
    orig_resampler_file = resampler_file.parent / ("orig_" + resampler_file.name)
    if not orig_resampler_file.exists():
        resampler_file.rename(orig_resampler_file)
        with orig_resampler_file.open("r") as f:
            content = f.read()
            content = content.replace("from typing import Tuple", "from typing import Tuple, List")
            with resampler_file.open("w") as out_f:
                out_f.write(content)


def convert_llm(model, model_dir):
    model.llm.config.save_pretrained(model_dir / text_emb_path.parent)
    if not (model_dir / text_emb_path).exists():
        print("⌛ Convert Text Embedding model")
        ov_model = ov.convert_model(model.llm.model.embed_tokens, example_input=torch.ones([1, 10], dtype=torch.long))

        ov.save_model(ov_model, model_dir / text_emb_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Text Embedding model successfully converted")

    if not (model_dir / llm_path).exists():
        print("⌛ Convert Language model")
        hidden_size = model.llm.config.hidden_size
        num_pkv = model.llm.config.num_hidden_layers
        pkv_shape = (2, model.llm.config.num_key_value_heads, 2, hidden_size // model.llm.config.num_attention_heads)

        input_embeds = torch.randn((2, 2, hidden_size))
        attention_mask = torch.ones([2, 4], dtype=torch.long)
        position_ids = torch.tensor([[2, 3], [2, 3]], dtype=torch.long)
        input_names = ["attention_mask", "position_ids"]
        output_names = ["logits"]

        past_key_values = []
        for i in range(num_pkv):
            kv = [torch.randn(pkv_shape) for _ in range(2)]
            past_key_values.append(kv)
            input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        input_names.append("inputs_embeds")

        example_input = {"inputs_embeds": input_embeds, "attention_mask": attention_mask, "position_ids": position_ids, "past_key_values": past_key_values}

        def forward_wrap(
            self,
            attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
        ):
            from transformers.cache_utils import DynamicCache

            if past_key_values is not None:
                pkv = DynamicCache.from_legacy_cache(past_key_values)
            result = self._orig_forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=pkv,
                inputs_embeds=inputs_embeds,
            )
            return (result.logits, result.past_key_values.to_legacy_cache())

        model.llm._orig_forward = model.llm.forward
        model.llm.forward = types.MethodType(forward_wrap, model.llm)

        ov_model = ov.convert_model(model.llm, example_input=example_input)

        for out, out_name in zip(ov_model.outputs, output_names):
            out.get_tensor().set_names({out_name})

        for inp, inp_name in zip(ov_model.inputs, input_names):
            inp.get_tensor().set_names({inp_name})

        patch_stateful(ov_model)

        shutil.copy(model_dir / "ckpt" / "configuration_minicpm.py", model_dir / "language_model" / "configuration_minicpm.py")
        shutil.copy(model_dir / "ckpt" / "modeling_navit_siglip.py", model_dir / "language_model" / "modeling_navit_siglip.py")
        ov.save_model(ov_model, model_dir / llm_path)
        del ov_model

        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Language model successfully converted")


def convert_vision_encoder(model, model_dir):
    tgt_sizes = torch.tensor([[24, 43]])
    if not (model_dir / image_encoder_path).exists():
        print("⌛ Convert Image Encoder model")

        def siglip_vis_embed_forward(
            self,
            pixel_values: torch.FloatTensor,
            patch_attention_mask: torch.BoolTensor,
            tgt_sizes: Optional[torch.IntTensor] = None,
            position_ids: Optional[torch.FloatTensor] = None,
        ) -> torch.Tensor:
            patch_embeds = self.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            if position_ids is None:
                batch_size = pixel_values.size(0)
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
                    if tgt_sizes is not None:
                        nb_patches_h = tgt_sizes[batch_idx][0]
                        nb_patches_w = tgt_sizes[batch_idx][1]
                    else:
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

        def siglip_attn_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
            """Input shape: Batch x Time x Channel"""

            batch_size, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attention_mask, is_causal=attention_mask is None
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

            attn_output = self.out_proj(attn_output)

            return attn_output, None

        def siglip_transformer_forward(
            self,
            pixel_values,
            patch_attention_mask: Optional[torch.BoolTensor] = None,
            tgt_sizes: Optional[torch.IntTensor] = None,
            position_ids: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[tuple, BaseModelOutputWithPooling]:
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

            hidden_states = self.embeddings(
                pixel_values=pixel_values, patch_attention_mask=patch_attention_mask, tgt_sizes=tgt_sizes, position_ids=position_ids
            )

            patch_attention_mask = patch_attention_mask.view(batch_size, -1)
            attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype) if not self._use_flash_attention_2 else patch_attention_mask

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.post_layernorm(last_hidden_state)

            if not return_dict:
                return (last_hidden_state, None) + encoder_outputs[1:]

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=None,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        vpm = model.vpm
        vpm.embeddings.forward = types.MethodType(siglip_vis_embed_forward, vpm.embeddings)
        for layer in vpm.encoder.layers:
            layer.self_attn.forward = types.MethodType(siglip_attn_forward, layer.self_attn)
        vpm.forward = types.MethodType(siglip_transformer_forward, vpm)

        pixel_values = torch.randn([1, 3, 14, 14448])
        patch_attn_mask = torch.zeros((1, 1, 1032), dtype=torch.bool)
        patch_attn_mask[0, 0, : tgt_sizes[0][0] * tgt_sizes[0][1]] = True
        position_ids = prepare_vis_position_ids(
            pixel_values, patch_attn_mask, tgt_sizes, model.config.vision_config.patch_size, model.config.vision_config.image_size // model.config.patch_size
        )
        ov_model = ov.convert_model(vpm, example_input={"pixel_values": pixel_values, "position_ids": position_ids, "patch_attention_mask": patch_attn_mask})
        ov.save_model(ov_model, model_dir / image_encoder_path)
        del ov_model
        cleanup_torchscript_cache()
        # del vpm
        # del model.vpm
        gc.collect()
        print("✅ Image Encoder model successfully converted")

    if not (model_dir / resampler_path).exists():
        print("⌛ Convert Resamler model")

        def resampler_forward(self, x, pos_embed, key_padding_mask):
            bs = x.shape[0]
            x = self.kv_proj(x)  # B * L * D
            x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

            q = self.ln_q(self.query)  # Q * D

            q_bs = q.unsqueeze(1).repeat(1, bs, 1)

            out = self.attn(q_bs, x + pos_embed, x, key_padding_mask=key_padding_mask)[0]  # Q * B * D  # L * B * D +  L * B * D
            #  out: Q * B * D
            x = out.permute(1, 0, 2)  # B * Q * D

            x = self.ln_post(x)
            x = x @ self.proj
            return x

        model.resampler.forward = types.MethodType(resampler_forward, model.resampler)

        pos_embed_base = get_2d_sincos_pos_embed(model.resampler.embed_dim, 70)

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((1, max_patch_len), dtype=torch.bool)

        pos_embed = []
        tgt_h, tgt_w = tgt_sizes[0]
        pos_embed = torch.from_numpy(pos_embed_base[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, 1, -1)))  # patches * D
        key_padding_mask[0, patch_len:] = True

        ov_model = ov.convert_model(model.resampler, example_input=[torch.randn(1, 1032, 1152), pos_embed, key_padding_mask])
        ov.save_model(ov_model, model_dir / resampler_path)
        del ov_model
        cleanup_torchscript_cache()
        del model.resampler
        gc.collect()
        print("✅ Resampler model successfully converted")


def convert_audio_encoder(model, model_dir):
    if not (model_dir / audio_encoder_path).exists():

        def forward_wrap(self, input_features, attention_mask):
            audio_states = self._orig_forward(input_features, output_hidden_states=True, attention_mask=attention_mask).hidden_states[self.audio_encoder_layer]
            audio_embeds = self.audio_projection_layer(audio_states)
            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)
            return audio_embeds

        model._orig_forward = model.apm.forward
        model.forward = types.MethodType(forward_wrap, model)

        print("⌛ Convert Audio Encoder model")
        input_features = torch.randn([2, 80, 1515])
        attention_mask = torch.ones([2, 1, 758, 758])
        ov_model = ov.convert_model(model, example_input=[input_features, attention_mask])
        ov.save_model(ov_model, model_dir / audio_encoder_path)
        del ov_model
        cleanup_torchscript_cache()
        print("✅ Audio Encoder model successfully converted")


def convert_minicpmo26(model_id, remove_checkpoint=False):
    model_dir = Path(model_id.split("/")[-1])
    requires_conversion = not all(
        [
            (model_dir / text_emb_path).exists(),
            (model_dir / image_encoder_path).exists(),
            (model_dir / resampler_path).exists(),
            (model_dir / llm_path).exists(),
        ]
    )

    if not requires_conversion:
        print(f"✅ {model_id} model already converted. You can find results in {model_dir}")
        return model_dir

    print(f"⌛ {model_id} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")
    ckpt = model_dir / "ckpt"
    if not ckpt.exists():
        snapshot_download(model_id, local_dir=ckpt, force_download=True)
        patch_model_code(ckpt)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="sdpa",  # sdpa or flash_attention_2
        torch_dtype=torch.float32,
        init_vision=True,
        init_audio=True,
    )
    print("✅ Original model successfully loaded")
    model.eval()
    model.config.save_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.save_pretrained(model_dir)

    convert_llm(model, model_dir)
    del model.llm
    gc.collect()
    convert_vision_encoder(model, model_dir)
    convert_audio_encoder(model, model_dir)
    print(f"✅ {model_id} model sucessfully converted. You can find results in {model_dir}")
    return model_dir


def prepare_vis_position_ids(pixel_values, patch_attention_mask, tgt_sizes, patch_size, num_patches_per_side):
    batch_size = pixel_values.size(0)
    max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
    max_nb_patches_h, max_nb_patches_w = max_im_h // patch_size, max_im_w // patch_size
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
    position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)

    for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
        if tgt_sizes is not None:
            nb_patches_h = tgt_sizes[batch_idx][0]
            nb_patches_w = tgt_sizes[batch_idx][1]
        else:
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

        pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
        position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

    return position_ids


core = ov.Core()


class OVModelForCausalLMWithEmb(GenerationMixin):
    def __init__(self, model_dir, device="CPU", ov_config=None, compile=True, slice_lm_head=True) -> None:
        self._supports_cache_class = False
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False
        self.generation_config = GenerationConfig.from_model_config(self.config)
        model_dir = Path(model_dir)
        self.model = core.read_model(model_dir / "language_model.xml")
        self.token_emb = core.read_model(model_dir / "embed_tokens.xml")
        if slice_lm_head:
            self.slice_lm_head()
        self.request = None
        self.token_emb_request = None
        self._device = device.upper()
        self.device = torch.device("cpu")
        self.ov_config = ov_config
        self.next_beam_idx = None
        self._past_length = None
        self.input_names = [input_t.get_any_name() for input_t in self.model.inputs]
        self.main_input_name = "input_ids"
        self.llm_times = []
        if compile:
            self.compile()

    def slice_lm_head(self):
        manager = Manager()
        manager.register_pass(InsertSlice())
        manager.run_passes(self.model)
        self.model.validate_nodes_and_infer_types()

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
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        inputs = {}
        # past_key_values are not used explicitly, instead they are handled inside the model
        if past_key_values is None:
            self.llm_times = []
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
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
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

        start = time.perf_counter()
        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        self.llm_times.append(time.perf_counter() - start)
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

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds if past_key_values is None else None,
        }

        return model_inputs

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        return self._past_length

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    def _reorder_cache(self, past_key_values: tuple[tuple[torch.Tensor]], beam_idx: torch.Tensor) -> tuple[tuple[torch.Tensor]]:
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


class OVMiniCPMO:
    def __init__(self, config, vpm, resampler, apm, llm, processor):
        self.config = config
        self.llm = llm
        self.vpm = vpm
        self.apm = apm
        self.embed_dim = self.llm.config.hidden_size
        self._resampler = resampler
        self.processor = processor
        self._pos_embeds = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, 70)).float()
        self.max_size = (70, 70)
        self.vpm_times = []
        self.resampler_times = []

        self.terminators = ["<|im_end|>", "<|endoftext|>"]

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def resampler(self, x, tgt_sizes):
        bs = x.shape[0]

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self._pos_embeds[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)))  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D

        start = time.perf_counter()
        res = torch.from_numpy(self._resampler([x, pos_embed, key_padding_mask])[0])
        self.resampler_times.append(time.perf_counter() - start)
        return res

    def _set_2d_pos_cache(self, max_size):
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float()
        self._pos_embed = pos_embed

    def _adjust_pos_cache(self, tgt_sizes):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size)

    def subsequent_chunk_mask(
        self,
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        """Create mask for subsequent steps (size, size) with chunk size,
        this is for streaming encoder

        Args:
            size (int): size of mask
            chunk_size (int): size of chunk
            num_left_chunks (int): number of left chunks
                <0: use full chunk
                >=0: use num_left_chunks
            device (torch.device): "cpu" or "cuda" or torch.Tensor.device

        Returns:
            torch.Tensor: mask

        Examples:
            >>> subsequent_chunk_mask(4, 2)
            [[1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]
        """
        ret = torch.zeros(size, size, device=device, dtype=torch.bool)
        for i in range(size):
            if num_left_chunks < 0:
                start = 0
            else:
                start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
            ending = min((i // chunk_size + 1) * chunk_size + num_lookhead, size)
            ret[i, start:ending] = True
        return ret

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (input_lengths_after_cnn - self.config.audio_pool_step) // self.config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_vllm_embedding(self, data):
        if "vision_hidden_states" not in data:
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

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = 32
                all_pixel_values = all_pixel_values
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        block_pxl_values = all_pixel_values[start_idx:end_idx]
                        block_patch_attn_mask = patch_attn_mask[start_idx:end_idx]
                        block_tgt_sizes = tgt_sizes[start_idx:end_idx]
                        block_position_ids = prepare_vis_position_ids(
                            block_pxl_values,
                            block_patch_attn_mask,
                            block_tgt_sizes,
                            self.config.vision_config.patch_size,
                            self.config.vision_config.image_size // self.config.patch_size,
                        )
                        start = time.perf_counter()
                        tmp_hs = torch.from_numpy(self.vpm([block_pxl_values, block_patch_attn_mask, block_position_ids])[0])
                        self.vpm_times.append(time.perf_counter() - start)
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    position_ids = prepare_vis_position_ids(
                        all_pixel_values,
                        patch_attn_mask,
                        tgt_sizes,
                        self.config.vision_config.patch_size,
                        self.config.vision_config.image_size // self.config.patch_size,
                    )
                    start = time.perf_counter()
                    vision_embedding = torch.from_numpy(self.vpm([all_pixel_values, patch_attn_mask, position_ids])[0])
                    self.vpm_times.append(time.perf_counter() - start)
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
                dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data["vision_hidden_states"]

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.embed_tokens(data["input_ids"])

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = torch.from_numpy(vllm_embedding[i])
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack([torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound])

                    cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]), cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
        return vllm_embedding, vision_hidden_states

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True):
        r"""
        Extract full audio embeddings with optional chunk-based attention.

        This method computes embeddings for all audio frames at once, either using full attention (when
        `chunk_length` is -1) or chunk-based attention (when `chunk_length` is a positive number). It does
        not use key-value caching and is suitable for non-streaming inference.

        Args:
            data (dict):
                - **"audio_features"** (`torch.FloatTensor`): Input mel-spectrograms of shape `(batch_size, 80, frames)`.
                - **"audio_feature_lens"** (list[list[int]]): Lengths of each audio segment for each item in the batch.
            chunk_length (int, optional): Determines whether to use full attention (-1) or chunk-based
                attention (>0) during embedding computation.

        Returns:
            list[list[torch.Tensor]]: audio embeddings
        """

        wavforms = data.get("audio_features", [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = data.get("audio_feature_lens", [])  # list, [[x1, x2], [y1], [z1]]
        # exist audio
        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feature_lens.dtype, device=audio_feature_lens.device).unsqueeze(0).expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand  # 1 for padded values

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(batch_size, 1, max_seq_len, max_seq_len)
            audio_attention_mask = audio_attention_mask_.to(device="cpu")

            if chunk_length > 0:
                chunk_num_frame = int(chunk_length * 50)
                chunk_mask = self.subsequent_chunk_mask(
                    size=max_seq_len,
                    chunk_size=chunk_num_frame,
                    num_left_chunks=-1,
                    device=audio_attention_mask_.device,
                )
                audio_attention_mask_ = torch.logical_or(audio_attention_mask_, torch.logical_not(chunk_mask))

            audio_attention_mask[audio_attention_mask_] = float("-inf")
            audio_outputs = self.apm([wavforms, audio_attention_mask])

            audio_embeds = torch.from_numpy(audio_outputs[-1])

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

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

    def get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
        """
        Args:
            data:
            input_embeddings:
            chunk_length: whisper use full attention or chunk attention
            stream_input: use streaming audio embedding
        Returns:
            final embeddings with audio feature
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

                if self.config.chunk_input:
                    for i in range(bs):
                        if not audio_embeddings[i]:
                            continue
                        audio_embs = torch.cat(audio_embeddings[i], dim=0)
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

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)

        if self.config.init_audio:
            vllm_embedding = self.get_omni_embedding(data, input_embeddings=vllm_embedding, chunk_length=self.config.audio_chunk_length)

        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        # compatible with llama factory
        for key in ["input_ids", "inputs_embeds", "position_ids"]:
            if key in kwargs:
                del kwargs[key]

        return self.llm(input_ids=None, position_ids=position_ids, inputs_embeds=vllm_embedding, **kwargs)

    def _decode(self, inputs_embeds, tokenizer, attention_mask, decode_text=False, **kwargs):
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict_in_generate", None)
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        output = self.llm.generate(
            inputs_embeds=torch.from_numpy(inputs_embeds),
            pad_token_id=0,
            eos_token_id=terminators,
            return_dict_in_generate=True,
            attention_mask=attention_mask,
            **kwargs,
        )
        return output

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {"inputs_embeds": torch.from_numpy(inputs_embeds), "pad_token_id": 0, "eos_token_id": terminators, "streamer": streamer}
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def get_sys_prompt(self, ref_audio=None, mode="default", language="zh"):
        """
        Choose different system prompts according to different tasks
        Args:
            ref_audio: if ref_audio is not None, will use the voice cloning prompts, and the voice
                       generated by the model will refer to the timbre of ref audio
            mode:
                "default": default system prompt and not refer to any task
                "omni": input video and audio simultaneously
                "audio_assistant": Default voice-only mode, the model will use the ref_audio's voice to reply user's question as a helpful assistant.
                "audio_roleplay": Roleplay voice-only mode, the model will use the ref_audio's voice to reply, and also role-play the character based on the audio prompt.
                "voice_cloning": TTS mode, the model will clone the voice of ref_audio.
            language: prompts language, the model has the ability to automatically select the response language
                    based on the question language
        Returns:

        """
        if ref_audio is not None:
            assert isinstance(ref_audio, np.ndarray), "ref_audio error"
        if mode == "omni":
            if language == "zh":
                sys_prompt = "你是一个AI助手。你能接受视频，音频和文本输入并输出语音和文本。"
                vc_prompt_prefix = sys_prompt + "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "作为助手，你将使用这种声音风格说话。"
            else:
                sys_prompt = "You are a helpful assistant. You can accept video, audio and text input and output voice and text. "
                vc_prompt_prefix = sys_prompt + "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "As an assistant, you will speak using this voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}

            else:
                sys_msgs = {"role": "user", "content": [sys_prompt]}

            return sys_msgs
        elif mode == "audio_assistant":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "作为助手，你将使用这种声音风格说话。"
            else:
                vc_prompt_prefix = "Use the voice in the audio prompt to synthesize new content."
                vc_prompt_suffix = "You are a helpful assistant with the above voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}

            else:
                logger.warning("Warning: ref_audio is None, speech generation will be performed based on the default voice.")
                sys_msgs = {"role": "user", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "audio_roleplay":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "假装你是上述音频中的人物，与我进行对话。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "Try to role-play the character based on the audio prompt above."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                print("Warning: ref_audio is None, speech generation will be performed based on the default voice.")
                sys_msgs = {"role": "user", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "voice_cloning":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio]}
            else:
                raise ValueError("ref_audio con't be None in voice_cloning mode.")

            return sys_msgs
        else:
            sys_prompt = "You are a helpful assistant. You can accept audio and text input and output voice and text."
            sys_msgs = {"role": "user", "content": [sys_prompt]}

            return sys_msgs

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        audio_features=[],
        audio_feature_lens=None,
        image_bound=None,
        audio_bounds=None,
        spk_bounds=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        stream=False,
        decode_text=True,
        **kwargs,
    ):
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

        model_output = {}
        with torch.inference_mode():
            model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
            model_inputs["inputs_embeds"] = self.get_omni_embedding(
                model_inputs,
                input_embeddings=model_inputs["inputs_embeds"],
                chunk_length=self.config.audio_chunk_length,
            )

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
                # if stream return TextIteratorStreamer and output is empty
                outputs = {}
            else:
                outputs = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, **kwargs)

                result = self._decode_text(outputs.sequences, tokenizer)

        if decode_text is False:
            return outputs

        return result, outputs

    def chat(
        self,
        image=None,
        msgs=None,
        tokenizer=None,
        processor=None,
        vision_hidden_states=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=32768,
        stream=False,
        chunk_input=True,
        omni_input=False,
        max_slice_nums=None,
        use_image_id=None,
        use_tts_template=False,
        generate_audio=False,
        return_spk_embed=False,
        return_dict=False,
        output_audio_path=None,
        **kwargs,
    ):
        """
        Unified chat function

        Args:
            image: use for batch_size=1 vqa, It is not recommended to continue to use this parameter
            msgs: the input chat msgs, support text: (string)  / image: (PIL.Image) / audio (numpy.ndarray)
            tokenizer: tokenizer for llm
            processor: if None, use the default processor
            max_new_tokens: the maximum length of the generation
            min_new_tokens: the minimum length of the generation
            sampling: whether to use sampling decoding or beam search decoding
            max_inp_length: the maximum length of input
            stream: whether to return generator, only used when tts is not required
            chunk_input: whether to split audio into 1s chunks
            omni_input: determine whether it is omni mode
            max_slice_nums: control the maximum number of image slices
            use_image_id: for video understanding or omni understanding, use_image_id should be False
            use_tts_template: if the msgs contain audio, use_tts_template should be True
            generate_audio: whether to generate audio output, only used when return_dict=True
            return_spk_embed: whether to return spk embedding, only used when return_dict=True
            return_dict: whether to return dict
            output_audio_path: audio save path when generate_audio
            **kwargs:
        """
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False

        if generate_audio or return_spk_embed:
            return_dict = True

        msgs_list = msgs
        images_list = image

        if batched is False:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        if processor is None:
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            processor = self.processor

        assert (
            self.config.query_num == processor.image_processor.image_feature_size
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.patch_size == processor.image_processor.patch_size
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.use_image_id == processor.image_processor.use_image_id
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.slice_mode == processor.image_processor.slice_mode
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        prompts_lists = []
        input_images_list = []
        input_audios_list = []
        audio_parts_list = []

        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert sampling or not stream, "if use stream mode, make sure sampling=True"

            if image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            audios = []
            audio_parts = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["system", "user", "assistant"]
                if i == 0:
                    assert role in ["user", "system"], "The role of first msg should be user"
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, np.ndarray):  # audio
                        audios.append(c)
                        audio_parts.append(i)
                        cur_msgs.append("(<audio>./</audio>)")
                        use_tts_template = True
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                if omni_input:
                    msg["content"] = "".join(cur_msgs)
                else:
                    msg["content"] = "\n".join(cur_msgs)

            prompts_lists.append(
                processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=None,
                )
            )
            input_images_list.append(images)
            input_audios_list.append(audios)
            audio_parts_list.append(audio_parts)

        inputs = processor(
            prompts_lists,
            input_images_list,
            input_audios_list,
            audio_parts_list,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            chunk_input=chunk_input,
            return_tensors="pt",
            max_length=max_inp_length,
        ).to("cpu")

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05,
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        if min_new_tokens > 0:
            generation_config["min_new_tokens"] = min_new_tokens

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        inputs.pop("image_sizes")
        with torch.inference_mode():
            res, outputs = self.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                **generation_config,
            )

        if stream:

            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, "")
                    yield text

            return stream_gen()

        else:
            spk_embeds = wav_numpy = sr = None

            if batched:
                answer = res
            else:
                answer = res[0]

                if use_tts_template and generate_audio:
                    mel_spec = self._generate_mel_spec(inputs, outputs, answer)
                    wav_numpy, sr = self.decode_mel_to_audio(mel_spec, output_audio_path)

            if return_spk_embed:
                spk_embeds = self._get_last_spk_embeds(inputs, outputs)

            if isinstance(answer, list):
                answer = [i.replace(tokenizer.tts_end, "") for i in answer]
            else:
                answer = answer.replace(tokenizer.tts_end, "")

            return answer


def init_model(model_dir, llm_model_dir, device):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    llm = OVModelForCausalLMWithEmb(model_dir / llm_model_dir, device)
    img_emb = core.compile_model(model_dir / image_encoder_path, device)
    aud_emb = core.compile_model(model_dir / audio_encoder_path, device)
    resampler = core.compile_model(model_dir / resampler_path, device)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    ov_model = OVMiniCPMO(config, img_emb, resampler, aud_emb, llm, processor)
    return ov_model


def lm_variant_selector(int4_model_dir):
    import ipywidgets as widgets

    use_int4_lang_model = widgets.Checkbox(value=int4_model_dir.exists(), description="INT4 language model", disabled=not int4_model_dir.exists())
    return use_int4_lang_model


def compression_widget(default_value=True):
    import ipywidgets as widgets

    to_compress_weights = widgets.Checkbox(
        value=default_value,
        description="Weights Compression",
        disabled=False,
    )

    return to_compress_weights
