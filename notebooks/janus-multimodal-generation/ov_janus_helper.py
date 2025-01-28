from pathlib import Path
import types
from typing import Optional, Tuple, List
import gc
import openvino as ov
from openvino.runtime import opset13
import nncf
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from janus.utils.io import load_pil_images
from PIL import Image

VISION_EMBEDDINGS = "openvino_vision_embeddings_model.xml"
TEXT_EMBEDDINGS = "openvino_text_embeddings_model.xml"
LANGUAGE_MODEL = "openvino_language_model.xml"
LM_HEAD = "openvino_lm_head_model.xml"
GEN_HEAD = "openvino_gen_head_model.xml"
GEN_EMBEDDINGS = "openvino_gen_embeddings_model.xml"
GEN_DECODER = "openvino_gen_decoder_model.xml"


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


def convert_janus_model(model_id, output_dir, quantization_config):
    model_name = Path(model_id).name
    output_dir = Path(output_dir)

    lang_model_path = output_dir / LANGUAGE_MODEL
    image_embed_path = output_dir / VISION_EMBEDDINGS
    lm_head_path = output_dir / LM_HEAD
    embed_token_path = output_dir / TEXT_EMBEDDINGS
    gen_head_path = output_dir / GEN_HEAD
    gen_embed_path = output_dir / GEN_EMBEDDINGS
    gen_decoder_path = output_dir / GEN_DECODER

    if all(
        [
            lang_model_path.exists(),
            image_embed_path.exists(),
            lm_head_path.exists(),
            embed_token_path.exists(),
            gen_head_path.exists(),
            gen_embed_path.exists(),
            gen_decoder_path.exists(),
        ]
    ):
        print(f"✅ {model_name} model already converted. You can find results in {output_dir}")
        return
    print(f"⌛ {model_name} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")
    processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    language_config = config.language_config
    language_config._attn_implementation = "sdpa"
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_id, language_config=language_config, trust_remote_code=True)
    vl_gpt = vl_gpt.eval()
    vl_gpt.config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("✅ Original model successfully loaded")
    hidden_size = vl_gpt.language_model.config.hidden_size
    if not embed_token_path.exists():
        print("⌛ Convert Input embedding model")
        ov_model = ov.convert_model(
            vl_gpt.language_model.get_input_embeddings(),
            example_input=torch.ones([2, 2], dtype=torch.int64),
        )
        ov.save_model(ov_model, embed_token_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Input embedding model successfully converted")

    if not lm_head_path.exists():
        print("⌛ Convert LM head model")
        ov_model = ov.convert_model(
            vl_gpt.language_model.lm_head,
            example_input=torch.ones([2, 2, hidden_size]),
        )
        ov.save_model(ov_model, lm_head_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ LM head model successfully converted")

    if not lang_model_path.exists():
        print("⌛ Convert Language model")

        language_model = vl_gpt.language_model.model

        def forward_wrap(
            self,
            attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
        ):
            from transformers.cache_utils import DynamicCache

            pkv = DynamicCache.from_legacy_cache(past_key_values)

            result = self._orig_forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=pkv,
                inputs_embeds=inputs_embeds,
            )
            key_values = result.past_key_values.to_legacy_cache()
            return (result[0], key_values)

        language_model._orig_forward = language_model.forward
        language_model.forward = types.MethodType(forward_wrap, language_model)
        hidden_size = language_model.config.hidden_size
        num_pkv = language_model.config.num_hidden_layers
        pkv_shape = (2, language_model.config.num_key_value_heads, 2, hidden_size // language_model.config.num_attention_heads)
        position_ids = torch.tensor([[2, 3], [2, 3]])

        input_embeds = torch.randn((2, 2, hidden_size))
        attention_mask = torch.ones([2, 4], dtype=torch.long)
        input_names = ["attention_mask", "position_ids"]
        output_names = ["last_hidden_state"]

        past_key_values = []
        for i in range(num_pkv):
            kv = [torch.randn(pkv_shape) for _ in range(2)]
            past_key_values.append(kv)
            input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        input_names.append("inputs_embeds")

        example_input = {"inputs_embeds": input_embeds, "attention_mask": attention_mask, "position_ids": position_ids, "past_key_values": past_key_values}

        ov_model = ov.convert_model(
            language_model,
            example_input=example_input,
        )

        for input, input_name in zip(ov_model.inputs, input_names):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, output_names):
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
        del language_model
        del vl_gpt.language_model
        gc.collect()
    if not image_embed_path.exists():
        print("⌛ Convert Image embedding model")
        from einops import rearrange

        def image_embedding_forward(self, pixel_values):
            bs, n = pixel_values.shape[0:2]
            images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
            # [b x n, T2, D]
            images_embeds = self.aligner(self.vision_model(images))

            # [b x n, T2, D] -> [b, n x T2, D]
            images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
            return images_embeds

        vl_gpt.forward = types.MethodType(image_embedding_forward, vl_gpt)

        ov_model = ov.convert_model(vl_gpt, example_input=torch.randn([1, 1, 3, 384, 384]))
        ov.save_model(ov_model, image_embed_path)
        del ov_model
        cleanup_torchscript_cache()
        del vl_gpt.aligner
        del vl_gpt.vision_model
        gc.collect()
        print("✅ Image embedding model successfully converted")

    if not gen_head_path.exists():
        print("⌛ Convert Gen head model")
        ov_model = ov.convert_model(
            vl_gpt.gen_head,
            example_input=torch.ones([2, hidden_size]),
        )
        ov.save_model(ov_model, gen_head_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Gen head model successfully converted")

    if not gen_embed_path.exists():
        print("⌛ Convert Gen image embeddings model")
        vl_gpt.forward = vl_gpt.prepare_gen_img_embeds
        ov_model = ov.convert_model(
            vl_gpt,
            example_input=torch.ones([2], dtype=torch.int64),
        )
        ov.save_model(ov_model, gen_embed_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Gen image embeddings model successfully converted")

    if not gen_decoder_path.exists():
        print("⌛ Convert Gen decoder model")
        dec = vl_gpt.gen_vision_model
        dec.forward = dec.decode_code
        ov_model = ov.convert_model(dec, example_input={"code_b": torch.ones([2, 576], dtype=torch.int64), "shape": torch.tensor([2, 8, 24, 24])})
        ov.save_model(ov_model, gen_decoder_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Gen decoder model successfully converted")
    del vl_gpt
    gc.collect()

    print(f"✅ {model_id} model conversion finished. You can find results in {output_dir}")


class OvModelForCausalLMWithEmb(GenerationMixin):
    def __init__(self, model_dir, device="CPU", ov_config=None, compile=True):
        self._supports_cache_class = False
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True).language_config
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False
        self.generation_config = GenerationConfig.from_model_config(self.config)
        model_dir = Path(model_dir)
        self.model = core.read_model(model_dir / "openvino_language_model.xml")
        self.token_emb = core.read_model(model_dir / "openvino_text_embeddings_model.xml")
        self.lm_head = core.read_model(model_dir / "openvino_lm_head_model.xml")
        self.request = None
        self.token_emb_request = None
        self.lm_head_request = None
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
        self._compile_lm_head()

    def _compile_token_emb(self):
        if self.token_emb_request is None:
            self.token_emb_request = core.compile_model(self.token_emb, self._device, self.ov_config)

    def _compile_lm_head(self):
        if self.lm_head_request is None:
            self.lm_head_request = core.compile_model(self.lm_head, self._device, self.ov_config)

    def to(self, device: str):
        if isinstance(device, str):
            self._device = device.upper()
            self.clear_requests()

        return self

    def clear_requests(self):
        del self.request
        del self.token_emb_request
        del self.lm_head_request
        self.request = None
        self.token_emb_request = None
        self.lm_head_request = None

    def embed_tokens(self, input_ids: torch.LongTensor):
        self._compile_token_emb()
        res = self.token_emb_request(input_ids, share_inputs=True)
        return res[0]

    def prepare_inputs(
        self,
        input_ids: Optional[torch.LongTensor],
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
                    position_ids = position_ids[:, -inputs_embeds.shape[1] :]

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
        hidden_state = self.request.get_tensor("last_hidden_state").data
        logits = self.lm_head_request(hidden_state, share_inputs=True, share_outputs=True)[0]
        logits = torch.from_numpy(logits).to(self.device)
        past_key_values = ((),)
        self._past_length += inputs["inputs_embeds"].shape[1]

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def model_forward(
        self,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        self.compile()

        inputs = self.prepare_inputs(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        hidden_state = self.request.get_tensor("last_hidden_state").data
        self._past_length += inputs["inputs_embeds"].shape[1]
        return hidden_state, ((),)

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


class OVJanusModel:
    def __init__(self, model_dir, device, ov_config=None):
        model_dir = Path(model_dir)
        self.language_model = OvModelForCausalLMWithEmb(model_dir, device, ov_config)
        self.vision_embeddings = core.compile_model(model_dir / "openvino_vision_embeddings_model.xml", device, ov_config)
        self.gen_embeddings = core.compile_model(model_dir / "openvino_gen_embeddings_model.xml", device, ov_config)
        self.gen_decoder = core.compile_model(model_dir / "openvino_gen_decoder_model.xml", device, ov_config)
        self.gen_head = core.compile_model(model_dir / "openvino_gen_head_model.xml", device, ov_config)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.LongTensor] = None,
        images_emb_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """

        Args:orch.cat([attention_mask, torch.zeros)
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """
        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = torch.from_numpy(self.language_model.embed_tokens(input_ids))

        if pixel_values is not None:
            images_embeds = torch.from_numpy(self.vision_embeddings(pixel_values)[0])

            # [b, n, T2] -> [b, n x T2]
            images_emb_mask = images_emb_mask.reshape(pixel_values.shape[0], -1)

            # replace with the image embeddings
            inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_embeddings(image_ids)[0]


def vl_conversation(ov_model, processor, input_prompt, image_path, streamer=None):
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>{input_prompt}\n",
            "images": [image_path],
        },
        {"role": "Assistant", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True)

    # # run image encoder to get the image embeddings
    inputs_embeds = ov_model.prepare_inputs_embeds(**prepare_inputs)

    generation_kwargs = {}

    if streamer:
        generation_kwargs["streamer"] = streamer

    # # run the model to get the response
    outputs = ov_model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
        **generation_kwargs,
    )

    answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


def generate_image(
    ov_model: OVJanusModel,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    output_dir: Path = Path("generated_samples"),
):
    conversation = [
        {
            "role": "User",
            "content": prompt,
        },
        {"role": "Assistant", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = ov_model.language_model.embed_tokens(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int)
    past_key_values = None

    for i in tqdm(range(image_token_num_per_image)):
        outputs = ov_model.language_model.model_forward(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
        hidden_states = torch.from_numpy(outputs[0])
        past_key_values = outputs[1]
        logits = torch.from_numpy(ov_model.gen_head(hidden_states[:, -1, :])[0])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = torch.from_numpy(ov_model.prepare_gen_img_embeds(next_token))
        inputs_embeds = img_embeds.unsqueeze(dim=1)
    dec = ov_model.gen_decoder([generated_tokens.to(dtype=torch.int), np.array([parallel_size, 8, img_size // patch_size, img_size // patch_size])])[0]
    dec = np.transpose(dec, (0, 2, 3, 1))

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    images = []

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    for i in range(parallel_size):
        images.append(Image.fromarray(visual_img[i]))
        if output_dir is not None:
            save_path = output_dir / f"img_{i}.jpg"
            images[-1].save(save_path)

    return images
