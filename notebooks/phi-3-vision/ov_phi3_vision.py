from pathlib import Path
import types
from typing import Optional, Tuple, Union, List
import gc
import openvino as ov
from openvino.runtime import opset13
import nncf
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast


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


core = ov.Core()


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def convert_phi3_model(model_id, output_dir, quantization_config):
    output_dir = Path(output_dir)

    lang_model_path = output_dir / "language_model.xml"
    image_embed_path = output_dir / "image_embed.xml"
    img_projection_path = output_dir / "img_projection.xml"
    embed_token_path = output_dir / "embed_token.xml"

    if all(
        [
            lang_model_path.exists(),
            image_embed_path.exists(),
            img_projection_path.exists(),
            embed_token_path.exists(),
        ]
    ):
        print(f"✅ Phi-3-vision model already converted. You can find results in {output_dir}")
        return
    print("⌛ Phi-3-vision conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, _attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("✅ Original model successfully loaded")

    if not embed_token_path.exists():
        print("⌛ Convert Input embedding model")
        ov_model = ov.convert_model(
            model.model.embed_tokens,
            example_input=torch.ones([2, 2], dtype=torch.int64),
        )
        ov.save_model(ov_model, embed_token_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Input embedding model successfully converted")

    vision_embed_tokens = model.model.vision_embed_tokens
    if not image_embed_path.exists():
        print("⌛ Convert Image embedding model")
        vision_embed_tokens.forward = vision_embed_tokens.get_img_features
        ov_model = ov.convert_model(vision_embed_tokens, example_input=torch.ones([17, 3, 336, 336]))
        ov.save_model(ov_model, image_embed_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Image embedding model successfully converted")

    if not img_projection_path.exists():
        print("⌛ Convert Image projection model")
        ov_model = ov.convert_model(
            vision_embed_tokens.img_projection,
            example_input=torch.ones([1, 1921, 4096]),
        )
        ov.save_model(ov_model, img_projection_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Image projection model successfully converted")

    if not lang_model_path.exists():
        print("⌛ Convert Language model")

        def forward_wrap(
            self,
            attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
        ):
            result = self._orig_forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
            )
            return tuple(result.values())

        model._orig_forward = model.forward
        model.forward = types.MethodType(forward_wrap, model)
        llm_input = torch.zeros([2, 2, 3072])
        pkv = model(
            inputs_embeds=llm_input,
            attention_mask=torch.ones((2, 2), dtype=torch.int64),
        )[1]
        model_inputs = ["attention_mask", "position_ids"]
        model_outputs = ["logits"]
        for idx in range(len(pkv)):
            model_inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
            model_outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])
        model_inputs.append("inputs_embeds")
        position_ids = torch.tensor([[2, 3], [2, 3]])
        ov_model = ov.convert_model(
            model,
            example_input={
                "inputs_embeds": llm_input,
                "attention_mask": torch.ones([2, 4], dtype=torch.int64),
                "past_key_values": pkv,
                "position_ids": position_ids,
            },
        )

        for input, input_name in zip(ov_model.inputs, model_inputs):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, model_outputs):
            output.get_tensor().set_names({output_name})
        patch_stateful(ov_model)
        print("✅ Language model successfully converted")

        if quantization_config is not None:
            print(f"⌛ Weights compression with {quantization_config['mode']} mode started")
            ov_model = nncf.compress_weights(ov_model, **quantization_config)
            print("✅ Weights compression finished")

        ov.save_model(ov_model, lang_model_path)
        del ov_model
        cleanup_torchscript_cache()
        del model
        gc.collect()
        print(f"✅ Phi-3-vision model conversion finished. You can find results in {output_dir}")


class OvPhi3Vision(GenerationMixin):
    def __init__(self, model_dir, device):
        model_dir = Path(model_dir)
        self.model = core.read_model(model_dir / "language_model.xml")
        self.image_embed = core.compile_model(model_dir / "image_embed.xml", device)
        self.img_projection = core.compile_model(model_dir / "img_projection.xml", device)
        self.embed_token = core.compile_model(model_dir / "embed_token.xml", device)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        compiled_model = core.compile_model(self.model, device)
        self.request = compiled_model.create_infer_request()
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self.num_pkv = 2
        self._supports_cache_class = False
        self.next_beam_idx = None
        self._past_length = None
        self.hd_transform_order = "glb_sub"
        self.num_img_tokens = self.config.img_processor["num_img_tokens"]
        self.image_dim_out = self.config.img_processor["image_dim_out"]
        self.glb_GN = torch.zeros([1, 1, self.image_dim_out * 4])
        self.sub_GN = torch.zeros([1, 1, 1, self.image_dim_out * 4])

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def __call__(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_sizes=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            image_sizes=image_sizes,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            if pixel_values is not None and image_sizes is not None:
                inputs_embeds = self.vision_embed_tokens(input_ids, pixel_values=pixel_values, image_sizes=image_sizes)
            else:
                inputs_embeds = self.embed_token(input_ids)[0]
        if past_key_values is None:
            self.request.reset_state()
            self.next_beam_idx = np.arange(inputs_embeds.shape[0], dtype=int)
            self._past_length = 0
        inputs = {}
        inputs["inputs_embeds"] = inputs_embeds
        inputs["attention_mask"] = attention_mask
        inputs["position_ids"] = position_ids
        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(inputs_embeds.shape[0], dtype=int)
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = self.request.get_tensor("logits").data
        logits = torch.from_numpy(logits).to(self.device)
        past_key_values = ((),)
        self._past_length += inputs["inputs_embeds"].shape[1]

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        return self._past_length

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = self._get_past_length(past_key_values)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_sizes": image_sizes,
            }
        )
        return model_inputs

    def vision_embed_tokens(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_sizes=None,
    ) -> torch.FloatTensor:
        MAX_INPUT_ID = int(1e9)
        img_embeds = pixel_values
        img_sizes = image_sizes

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        with torch.no_grad():
            positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=False)

        select = False
        if len(positions.tolist()) > 0:
            g_values = abs(input_ids[positions[:, 0], positions[:, 1]])

            if img_sizes is not None and len(img_sizes):
                hd_transform = True
                bs = img_embeds.shape[0]
                # Nx(HW)xC
                img_features = torch.from_numpy(self.image_embed(img_embeds.flatten(0, 1))[0])
                base_feat_height = base_feat_width = int(img_features.shape[1] ** 0.5)

                # bs x max_num_crops x (24x24) x C
                img_features = img_features.view(bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
                C = self.image_dim_out
                H = base_feat_height

                output_imgs = []
                output_len = []
                # training is tensor, inference is list
                if isinstance(img_sizes, torch.Tensor):
                    img_sizes = img_sizes.view(-1, 2)
                for _bs in range(bs):
                    h, w = img_sizes[_bs]
                    h = h // 336
                    w = w // 336
                    B_ = h * w

                    # 1 x (24x24) x 1024
                    global_img_feature = img_features[_bs, :1]

                    # 1 x 12 x 12 x 4096
                    glb_img = (
                        global_img_feature.reshape(1, H, H, C)
                        .reshape(1, H // 2, 2, H // 2, 2, C)
                        .contiguous()
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(1, H // 2, H // 2, 4 * C)
                        .contiguous()
                    )
                    temp_glb_GN = self.sub_GN.repeat(1, H // 2, 1, 1)

                    # 1 x 156 x 4096
                    glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1, -1, 4 * C)

                    # (max_num_crops-1) x (12x12) x C
                    sub_img = img_features[_bs, 1:]
                    # 16x574x1024
                    # get rid of padding sub_img
                    sub_img = sub_img[:B_]

                    # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
                    sub_img = (
                        sub_img.reshape(B_, H, H, C)
                        .reshape(B_, H // 2, 2, H // 2, 2, C)
                        .contiguous()
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(B_, -1, 4 * C)
                        .contiguous()
                    )
                    sub_img = sub_img.reshape(1, h, w, 12, 12, -1).permute(0, 1, 3, 2, 4, 5).reshape(1, h * 12, w * 12, 4 * C)
                    temp_sub_GN = self.sub_GN.repeat(1, h * 12, 1, 1)
                    sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1, -1, 4 * C)
                    # (1, num_img_tokens, 1024*4)

                    # glb + sub
                    if self.hd_transform_order == "glb_sub":
                        output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                    elif self.hd_transform_order == "sub_glb":
                        output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
                    else:
                        raise NotImplementedError(f"hd_transform_order = {self.hd_transform_order}, not implemented")

                    temp_len = int((h * w + 1) * 144 + 1 + (h + 1) * 12)
                    output_len.append(temp_len)

                num_img_tokens = output_len
                img_set_tensor = []
                for _output_img in output_imgs:
                    img_feature_proj = torch.from_numpy(self.img_projection(_output_img)[0])
                    img_set_tensor.append(img_feature_proj)
            elif img_embeds.ndim == 4:
                selected_g_values = g_values[:: self.num_img_tokens]
                tt = self.image_embed(img_embeds).reshape(-1, self.image_dim_out)[0]
                img_set_tensor = torch.from_numpy(self.img_projection(tt)[0])  # adapted visual features.
            elif img_embeds.ndim == 3:
                selected_g_values = g_values[:: self.num_img_tokens]
                tt = img_embeds.view(-1, self.image_dim_out)
                img_set_tensor = torch.from_numpy(self.img_projection(tt)[0])  # adapted visual features.
            else:
                raise NotImplementedError
            select = True
            input_ids.clamp_min_(0).clamp_max_(self.config.vocab_size)

        hidden_states = torch.from_numpy(self.embed_token(input_ids)[0])
        if select:
            if hd_transform:
                idx = 0
                for i, cnt in enumerate(num_img_tokens):
                    hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = img_set_tensor[i]
                    idx += cnt
            else:
                idx = 0
                for i, g in enumerate(selected_g_values):
                    cnt = self.num_img_tokens
                    hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = img_set_tensor[i * cnt : (i + 1) * cnt]
                    idx += cnt

        return hidden_states
