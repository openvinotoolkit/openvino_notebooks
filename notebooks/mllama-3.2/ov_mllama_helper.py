from pathlib import Path
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoConfig, GenerationConfig, TextStreamer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import repeat_kv
from typing import Optional, Union, List, Tuple, Dict
from optimum.exporters.openvino.stateful import patch_stateful
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
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


class InsertSlice(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Result")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            if root is None:
                return False
            root_output = matcher.get_match_value()
            root_name = root.get_friendly_name()
            if (len(root.get_output_partial_shape(0)) == 3):
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

        self.register_matcher(Matcher(param,"InsertSlice"), callback)


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

def convert_mllama(model_id, out_dir):

    out_dir = Path(out_dir)

    img_encoder_path = out_dir / IMAGE_ENCODER
    lang_model_path = out_dir / LANGUAGE_MODEL

    requires_conversion = not all([img_encoder_path.exists(), lang_model_path.exists()])
    if not requires_conversion:
        print(f"model already converted and can be found in {out_dir}")
        return
    print("Load original model")
    model = MllamaForConditionalGeneration.from_pretrained(model_id)
    model.config.save_pretrained(out_dir)
    model.generation_config.save_pretrained(out_dir)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.save_pretrained(out_dir)

    if not img_encoder_path.exists():
        print("Convert vision model...")

        class VisionEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask):
                bsz = pixel_values.shape[0]
                cross_attention_states = self.model.vision_model(pixel_values, aspect_ratio_ids, aspect_ratio_mask)
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
        ov_model = ov.convert_model(image_encoder, example_input={"pixel_values": torch.randn((1, 1, 4, 3, model.config.vision_config.image_size, model.config.vision_config.image_size)), "aspect_ratio_ids": torch.tensor([[6]]) , "aspect_ratio_mask": torch.tensor([[[1, 1, 1, 1]]])})

        output_names = ["cross_attention_states"]
        
        for i in model.config.text_config.cross_attention_layers:
            output_names.extend([f"cross_attn_key_values.{i}.key", f"cross_attn_key_values.{i}.value"])
        
        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})
        
        ov.save_model(ov_model, img_encoder_path)
        del ov_model
        cleanup_torchscript_cache()
        del image_encoder
        gc.collect()

        print("Vision model successfully converted")
    
    if not lang_model_path.exists():
        def lm_forward_wrapper(self, 
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cross_attention_mask: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            past_key_values: Optional[ List[torch.FloatTensor]] = None,
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
            result = self.orig_forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, cross_attention_mask=cross_attention_mask, full_text_row_masked_out_mask=full_text_row_masked_out_mask, past_key_values=common_cache, cache_position=cache_position)
            present_kv = [kv_cache for idx, kv_cache in enumerate(result.past_key_values.to_legacy_cache()) if idx not in self.model.cross_attention_layers]
            return result.logits, tuple(present_kv)
        
        model.language_model.orig_forward = model.language_model.forward
        model.language_model.forward = types.MethodType(lm_forward_wrapper, model.language_model)
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
        

        ov_model = ov.convert_model(model.language_model, example_input=example_inpit)

        for input, input_name in zip(ov_model.inputs, input_names):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})
        
        patch_stateful(model.config.text_config, ov_model)
        ov.save_model(ov_model, lang_model_path)
        del ov_model
        cleanup_torchscript_cache()
        del model
        gc.collect()

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
    def __init__(self, model_dir:Union[str, Path],
                 device:str="CPU",
                 ov_config:Optional[Dict[str, str]]=None,
                 language_model_name=None, image_encoder_name=None, slice_lm_head=True, use_remote_tensors=True, dynamic_shape=False):
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
        self.update_pkv_precision()
        if slice_lm_head:
            self.slice_lm_head()
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.lm_cross_attn_inputs = [key for key in self.input_names if "cross_attn_key_values" in key]
        compiled_model = core.compile_model(self.model, device, ov_config)
        self.request = compiled_model.create_infer_request()
        self.cross_attn_outputs = [key.get_any_name() for key in self.vision_model.outputs if "cross_attn_key_values" in key.get_any_name() ]
        compiled_vision_model = core.compile_model(self.vision_model, device, ov_config)
        self.vision_request = compiled_vision_model.create_infer_request()
        self.use_remote_tensors = use_remote_tensors
        if self._device == "GPU" and use_remote_tensors:
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
        self.vision_model.reshape({0: ov.PartialShape([1, 1, 4, 3, self.config.vision_config.image_size, self.config.vision_config.image_size]), 1: ov.PartialShape([1, 1]), 2: ov.PartialShape([1, 1, 4])})

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
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            start = time.perf_counter()
            self.vision_request.start_async([pixel_values, aspect_ratio_ids, aspect_ratio_mask], share_inputs=True)
            self.vision_request.wait()
            end = time.perf_counter()
            self.vision_encoder_infer_time.append(end - start)

            cross_attn_key_values = [self.vision_request.get_tensor(name) for name in self.cross_attn_outputs]
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
            cross_attention_key_values=cross_attn_key_values
        )

    def language_model(self, input_ids, attention_mask, position_ids, cross_attention_mask, full_text_row_masked_out_mask, past_key_values, cache_position, cross_attention_key_values):
        model_inputs = {
            "input_ids": ov.Tensor(np.array(input_ids)),
            "attention_mask": ov.Tensor(np.array(attention_mask)),
            "position_ids": ov.Tensor(np.array(position_ids)),
            "cross_attention_mask": ov.Tensor(np.array(cross_attention_mask)),
            "full_text_row_masked_out_mask": ov.Tensor(np.array(full_text_row_masked_out_mask)),
            "cache_position": ov.Tensor(np.array(cache_position))
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
        past_key_values = ((), )
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

        # TODO: we have no attention_mask so this won't work, check if we really won't need attention mask and find another way
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
                "cross_attn_key_values": cross_attn_key_values
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
            model_kwargs["cross_attention_mask"] = torch.cat(
                [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
            )
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
        cross_attention_mask = inverted_cross_attn_mask.masked_fill(
            inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
        )

        # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
        # last dimension contains negative infinity values, otherwise it's 1
        negative_inf_value = torch.finfo(dtype).min
        full_text_row_masked_out_mask = (
            (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
        )
        cross_attention_mask *= full_text_row_masked_out_mask

        # In case we receive a new image but already have previous cross-attention key/values in cache,
        # then we need to extend the attention-mask and add previous images' lengths
        if (
            past_key_values is not None
            and cross_attention_states is not None
            and cross_attention_layers is not None
        ):
            # make all zeros mask for cross-attn-mask from previuos cached hidden_states, all zeros right?
            # i.e. extend current cross-attn-mask on image-seq-length dimension to account for past_seen_tokens
            past_cross_attn_kv_length = cross_attention_layers[0].shape[-2]
            past_cross_attn_mask = torch.zeros(
                (*cross_attention_mask.shape[:-1], past_cross_attn_kv_length), dtype=dtype, device=device
            )
            # concatenate both on image-seq-length dimension
            cross_attention_mask = torch.cat([past_cross_attn_mask, cross_attention_mask], dim=-1)

        return cross_attention_mask, full_text_row_masked_out_mask

    def prepare_vision_outputs(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask,
                           cross_attention_mask=None, past_key_values=None, cache_position=None):
        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            self.vision_request.start_async([pixel_values, aspect_ratio_ids, aspect_ratio_mask], share_inputs=True)
            self.vision_request.wait()
            cross_attn_key_values = [self.vision_request.get_tensor(name).data for name in self.cross_attn_outputs]
            cross_attention_states = torch.from_numpy(self.vision_request.get_tensor("cross_attention_states").data)
        cross_attention_mask, full_text_row_masked_out_mask = self._prepare_cross_attention_mask(
            cross_attention_mask,
            past_key_values=past_key_values,
            num_vision_tokens=self.num_patches,
            cross_attention_layers=cross_attn_key_values if past_key_values is not None else None,
            cross_attention_states=cross_attention_states,
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
            "cross_attention_key_values": cross_attn_key_values
        }

    def prepare_llm_inputs(self, input_ids, attention_mask, position_ids, cross_attention_mask, full_text_row_masked_out_mask, past_key_values, cache_position, cross_attention_key_values):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cross_attention_mask": cross_attention_mask,
            "full_text_row_masked_out_mask": full_text_row_masked_out_mask,
            "cache_position": cache_position
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
    #convert_mllama("/home/ea/llama3.2/Llama-3.2-11B-Vision-Instruct", "Llama-3.2-11B-Vision-Instruct/OV")
    model_id = "Llama-3.2-11B-Vision-Instruct/OV"
    LANGUAGE_MODEL_NAME = "llm_int4_asym_r10_gs64_max_activation_variance_all_layers.xml"
    IMAGE_ENCODER_NAME = "openvino_vision_encoder.xml"
    ov_model = OVMLlamaForConditionalGeneration(model_id, device="GPU", language_model_name=LANGUAGE_MODEL_NAME, image_encoder_name=IMAGE_ENCODER_NAME)
    processor = AutoProcessor.from_pretrained(model_id)
    url = "https://llava-vl.github.io/static/images/view.jpg"
    raw_image = Image.open(requests.get(url, stream=True).raw)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe image in two sentences"}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    url = "https://llava-vl.github.io/static/images/view.jpg"
    raw_image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=text, images=raw_image, return_tensors="pt")
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    output = ov_model.generate(**inputs, do_sample=False, max_new_tokens=50, streamer=streamer)
    print(f"Visual encoder time {ov_model.vision_encoder_infer_time[0]}ms")
    print(f"First token latency {ov_model.llm_infer_time[0] * 1000}ms, Second token latency {np.mean(np.array(ov_model.llm_infer_time[1:])) * 1000}ms")
