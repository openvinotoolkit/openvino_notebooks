from pathlib import Path
import types
from typing import Optional, Tuple, Union, List
import gc

import openvino as ov
from openvino.runtime import opset13
import nncf
import numpy as np
import torch
from transformers.cache_utils import Cache
from transformers import (
    AutoModelForCausalLM,
    AutoImageProcessor,
    AutoConfig,
    AutoTokenizer,
)
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from typing import Optional, Tuple, Union, List, Dict, Any
from transformers import __version__ as transformers_version
from transformers.generation.utils import GenerationConfig, ModelOutput


def _glmv_transformer_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    **loss_kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    """take care of image_encode, position_ids and (attention_mask = None is fine)"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
    logits = logits.to(torch.float32)
    # if not return_dict:
    output = (logits,) + outputs[1:]
    return output


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


def convert_glmv_model(model_id, output_dir, quantization_config):
    model_name = Path(model_id).name
    output_dir = Path(output_dir)

    lang_model_path = output_dir / "openvino_language_model.xml"
    image_embed_path = output_dir / "openvino_vision.xml"
    embed_token_path = output_dir / "openvino_embedding.xml"
    ori_token_config_path = Path(model_id) / "tokenizer_config.json"
    ov_token_config_path = output_dir / "tokenizer_config.json"
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    image_size = config.vision_config["image_size"]

    if all(
        [
            lang_model_path.exists(),
            image_embed_path.exists(),
            embed_token_path.exists(),
        ]
    ):
        print(f"✅ {model_name} model already converted. You can find results in {output_dir}")
        return
    print(f"⌛ {model_name} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        _attn_implementation="eager",
    )
    model.config.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.save_pretrained(output_dir)
    # shutil.copy2(ori_token_config_path, ov_token_config_path)

    print("✅ Original model successfully loaded")

    if not embed_token_path.exists():
        print("⌛ Convert Input embedding model")
        ov_model = ov.convert_model(
            model.model.embed_tokens,
            example_input=torch.ones([1, 10], dtype=torch.int64),
        )
        ov.save_model(ov_model, embed_token_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Input embedding model successfully converted")

    if not image_embed_path.exists():
        print("⌛ Convert Image embedding model")
        # vision_embed_tokens.forward = vision_embed_tokens.vit
        ov_model = ov.convert_model(model.model.vision, example_input=torch.ones([1, 3, image_size, image_size]))
        ov.save_model(ov_model, image_embed_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Image embedding model successfully converted")

    if not lang_model_path.exists():
        print("⌛ Convert Language model")

        input_ids = torch.zeros([2, 2], dtype=torch.int64)
        inputs_embeds = torch.zeros([2, 2, config.hidden_size], dtype=torch.float32)

        pkv = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones((2, 2), dtype=torch.int64),
        )[1]
        model.forward = types.MethodType(_glmv_transformer_forward, model)

        model.config.torchscript = True
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
                "position_ids": position_ids,
                "inputs_embeds": inputs_embeds,
                "attention_mask": torch.ones([2, 4], dtype=torch.int64),
                "past_key_values": pkv,
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
        print(f"✅ {model_name} model conversion finished. You can find results in {output_dir}")


def is_empty(images_list: Optional[List[List[torch.Tensor]]]):
    if images_list is None or len(images_list) == 0:
        return True
    for image_list in images_list:
        if image_list is not None:
            return False
    return True


class OvGLMv(GenerationMixin):
    def __init__(self, model_dir, device):
        model_dir = Path(model_dir)
        self.model = core.read_model(model_dir / "openvino_language_model.xml")
        self.vision = core.compile_model(model_dir / "openvino_vision.xml", "CPU")
        self.embedding = core.compile_model(model_dir / "openvino_embedding.xml", "CPU")
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        # compiled_model = core.compile_model(self.model, device, config={"GPU_ENABLE_SDPA_OPTIMIZATION": "NO", "INFERENCE_PRECISION_HINT": "FP32"})
        compiled_model = core.compile_model(self.model, device)

        self.request = compiled_model.create_infer_request()
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self.num_pkv = 2
        self._supports_cache_class = False
        self.next_beam_idx = None
        self.hd_transform_order = "glb_sub"

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.Tensor = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.Tensor = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        if not past_key_values:
            self.request.reset_state()
            self.next_beam_idx = np.arange(input_ids.shape[0], dtype=int)
            # not allow for inputs_embeds, because we want to process image feature
            assert input_ids is not None and inputs_embeds is None, f"{input_ids} {inputs_embeds}"
            inputs_embeds = torch.from_numpy(self.embedding(input_ids)[0])
            new_input_embeds = []
            multi_flags = [True if self.config.boi_token_id in input_id.tolist() else False for input_id in input_ids]
            images_features = None
            if not is_empty(pixel_values):
                images_features = torch.from_numpy(self.vision(pixel_values)[0])
            image_count = 0
            for i in range(len(input_ids)):
                input_id = input_ids[i].tolist()
                if multi_flags[i]:
                    boi_token_pos = input_id.index(self.config.boi_token_id)
                    assert boi_token_pos >= 0, "begin_of_image not found!"
                    num_image_padding_tokens = input_id.count(self.config.boi_token_id)
                    assert num_image_padding_tokens == images_features[image_count].shape[0], f"Wrong image padding token number: {num_image_padding_tokens}"
                    new_input_embeds.append(
                        torch.cat(
                            (
                                inputs_embeds[i, :boi_token_pos],
                                images_features[image_count].to(inputs_embeds.device),
                                inputs_embeds[i, boi_token_pos + num_image_padding_tokens :],
                            )
                        )
                    )
                    image_count += 1
                else:
                    new_input_embeds.append(inputs_embeds[i])
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)[0]
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

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        if int(transformers_version.split(".")[1]) >= 44:
            assert not standardize_cache_format
            _, cache = self._extract_past_from_model_output(outputs)
            model_kwargs["past_key_values"] = cache
        else:
            cache = self._extract_past_from_model_output(outputs, standardize_cache_format)

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat([position_ids, new_position_id], dim=-1)

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = torch.zeros([1, 1, 1, 3, 672, 672]),
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        is_first_forward: bool = True,
        **kwargs,
    ) -> dict:
        if position_ids is None:
            if attention_mask is None:
                # Can only build sequential ids. Raise error right now
                raise ValueError("Cannot create position ids when attention mask is None")
            else:
                position_ids = self._create_position_ids_from_attention_mask(attention_mask)
        if not is_first_forward:
            if past_key_values is not None:
                position_ids = position_ids[..., -1:]
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def _create_position_ids_from_attention_mask(self, attention_mask):
        # Initialize a tensor of the same shape as attention_mask to hold position IDs
        position_ids = torch.zeros_like(attention_mask, dtype=torch.long, device=attention_mask.device)
        # Iterate over the batch
        for i, mask in enumerate(attention_mask):
            # Find the positions where the mask is 1
            positions = torch.nonzero(mask, as_tuple=False).squeeze(1).to(attention_mask.device)
            # Assign position IDs to those positions
            position_ids[i, positions] = torch.arange(start=0, end=positions.size(0), dtype=torch.long).to(attention_mask.device)
        return position_ids
