import gc
import types
from pathlib import Path
from typing import Optional, Tuple, List

import huggingface_hub as hf_hub
import openvino as ov
import numpy as np
import torch
from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM, GenerationMixin, GenerationConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

IMAGE_EMBEDDING_NAME = "image_embedding.xml"
TEXT_EMBEDING_NAME = "text_embedding.xml"
ENCODER_NAME = "encoder.xml"
DECODER_NAME = "decoder.xml"
DECODER_WITH_PAST_NAME = "decoder_with_past.xml"

core = ov.Core()

model_ids = ["microsoft/Florence-2-base-ft", "microsoft/Florence-2-base", "microsoft/Florence-2-large-ft", "microsoft/Florence-2-large"]


def get_model_selector(default="microsoft/Florence-2-base-ft"):
    import ipywidgets as widgets

    model_checkpoint = widgets.Dropdown(
        options=model_ids,
        default=default,
        description="Model:",
    )

    return model_checkpoint


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def download_original_model(model_id, orig_model_dir):
    hf_hub.snapshot_download(repo_id=model_id, local_dir=orig_model_dir)
    modeling_file = orig_model_dir / "modeling_florence2.py"
    orig_modeling_file = orig_model_dir / f"orig_{modeling_file.name}"
    if not orig_modeling_file.exists():
        modeling_file.rename(orig_modeling_file)
    with orig_modeling_file.open("r") as f:
        content = f.read()
        content = content.replace("if is_flash_attn_2_available():", "")
        content = content.replace("    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input", "")
        content = content.replace("    from flash_attn import flash_attn_func, flash_attn_varlen_func", "")
        content = content.replace("    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input", "")
        with modeling_file.open("w") as out_f:
            out_f.write(content)


def convert_florence2(model_id, output_dir, orig_model_dir=None):
    output_dir = Path(output_dir)

    required_conversion = not all(
        [
            (output_dir / IMAGE_EMBEDDING_NAME).exists(),
            (output_dir / TEXT_EMBEDING_NAME).exists(),
            (output_dir / ENCODER_NAME).exists(),
            (output_dir / DECODER_NAME).exists(),
            (output_dir / DECODER_WITH_PAST_NAME).exists(),
        ]
    )
    if not required_conversion:
        print(f"✅ {model_id} already converted and can be found in {output_dir}")
        return

    print(f"⌛ {model_id} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")
    if orig_model_dir is None:
        orig_model_dir = output_dir / "chkpt"
    if not orig_model_dir.exists():
        download_original_model(model_id, orig_model_dir)

    model = AutoModelForCausalLM.from_pretrained(orig_model_dir, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(orig_model_dir, trust_remote_code=True)
    processor.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    print("✅ Original model successfully loaded")
    if not (output_dir / IMAGE_EMBEDDING_NAME).exists():
        print("⌛ Image Embeddings conversion started")

        model._orig_forward = model.forward
        model.forward = model._encode_image

        example_input = torch.zeros([1, 3, model.config.vision_config.projection_dim, model.config.vision_config.projection_dim])

        ov_model = ov.convert_model(
            model, example_input=example_input, input=[-1, 3, model.config.vision_config.projection_dim, model.config.vision_config.projection_dim]
        )
        ov.save_model(ov_model, output_dir / IMAGE_EMBEDDING_NAME)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Image Embeddings successfuly converted")
        model.forward = model._orig_forward

    if not (output_dir / TEXT_EMBEDING_NAME).exists():
        print("⌛ Text Embedding conversion started")
        text_embedding = model.get_input_embeddings()
        example_input = torch.ones([2, 2], dtype=torch.long)
        ov_model = ov.convert_model(text_embedding, example_input=example_input)
        ov.save_model(ov_model, output_dir / TEXT_EMBEDING_NAME)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Text Embedding conversion started")

    if not (output_dir / ENCODER_NAME).exists():
        print("⌛ Encoder conversion started")
        encoder = model.get_encoder()
        example_input = {"inputs_embeds": torch.zeros([1, 590, model.config.text_config.d_model]), "attention_mask": torch.ones([1, 590])}
        ov_model = ov.convert_model(encoder, example_input=example_input)
        ov.save_model(ov_model, output_dir / ENCODER_NAME)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Encoder conversion finished")

    if not (output_dir / DECODER_NAME).exists() or not (output_dir / DECODER_WITH_PAST_NAME).exists():
        print("⌛ Decoder conversion started")
        model._orig_forward = model.forward

        def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask, past_key_values=None):
            result = self._orig_forward(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=(encoder_hidden_states,),
                attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
            )
            logits = result.logits
            pkv = result.past_key_values
            if past_key_values is not None:
                pkv = tuple([v[:2] for v in pkv])

            return logits, pkv

        model.forward = types.MethodType(forward, model)

        # decoder = model.get_decoder()
        # model.config.return_dict = False
        example_input = {
            "decoder_input_ids": torch.ones([1, 1], dtype=torch.long),
            # "attention_mask": torch.ones([1, 1], dtype=torch.long),
            "encoder_hidden_states": torch.ones([1, 590, model.config.text_config.d_model]),
            "encoder_attention_mask": torch.ones([1, 590], dtype=torch.long),
        }

        input_names = []
        output_names = ["logits"]
        past_key_values = []
        dec_attention_head = model.config.text_config.decoder_attention_heads
        enc_attention_head = model.config.text_config.encoder_attention_heads
        dec_hidden_dim = model.config.text_config.d_model // dec_attention_head
        enc_hidden_dim = model.config.text_config.d_model // enc_attention_head
        for i in range(model.config.text_config.decoder_layers):
            decoder_key = torch.randn([1, dec_attention_head, 1, dec_hidden_dim])
            decoder_value = torch.randn([1, dec_attention_head, 1, dec_hidden_dim])
            encoder_key = torch.randn([1, dec_attention_head, 590, enc_hidden_dim])
            encoder_value = torch.randn([1, dec_attention_head, 590, enc_hidden_dim])
            past_key_values.append((decoder_key, decoder_value, encoder_key, encoder_value))
            input_names.extend(
                [f"past_key_value.{i}.decoder.key", f"past_key_value.{i}.decoder.value", f"past_key_value.{i}.encoder.key", f"past_key_value.{i}.encoder.value"]
            )
            output_names.extend([f"present.{i}.decoder.key", f"present.{i}.decoder.value", f"present.{i}.encoder.key", f"present.{i}.encoder.value"])

        if not (output_dir / DECODER_NAME).exists():
            ov_model = ov.convert_model(model, example_input=example_input)
            for out, name in zip(ov_model.outputs, output_names):
                out.get_tensor().set_names({name})
            ov.save_model(ov_model, output_dir / DECODER_NAME)
            del ov_model
            cleanup_torchscript_cache()
            gc.collect()

        if not (output_dir / DECODER_WITH_PAST_NAME).exists():
            example_input["past_key_values"] = past_key_values
            # example_input["attention_mask"] = torch.ones([1, 2], dtype=torch.long)

            ov_model = ov.convert_model(model, example_input=example_input)
            for out, name in zip(ov_model.outputs, output_names):
                out.get_tensor().set_names({name})

            for inp, name in zip(ov_model.inputs[3:], input_names):
                inp.get_tensor().set_names({name})

            ov.save_model(ov_model, output_dir / DECODER_WITH_PAST_NAME)
            del ov_model
            cleanup_torchscript_cache()
            gc.collect()

        print("✅ Decoder conversion finished")
        model.forward = model._orig_forward
    print(f"✅ {model_id} already converted and can be found in {output_dir}")


class OVEncoder:
    """
    Encoder model for OpenVINO inference.

    Arguments:
        request (`openvino.runtime.ie_api.InferRequest`):
            The OpenVINO inference request associated to the encoder.
    """

    def __init__(self, model_dir, parent_model, device, ov_config):
        self.model = core.read_model(model_dir / ENCODER_NAME)
        self._device = device
        self.input_embedding = core.compile_model(model_dir / TEXT_EMBEDING_NAME, device, ov_config)
        self.parent_model = parent_model
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.input_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.inputs}
        self.main_input_name = "input_ids"
        compiled_model = core.compile_model(self.model, self._device, ov_config)
        self.request = compiled_model.create_infer_request()

    @property
    def device(self):
        return self.parent_model.device

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return torch.float32

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        **kwargs,
    ) -> BaseModelOutput:
        # Model inputs
        if input_ids is None and inputs_embeds is None:
            raise ValueError("`input_ids` or `inputs_embeds` should be provided")
        elif input_ids is not None:
            inputs_embeds = self.input_embedding(input_ids)[0]

        inputs = {"inputs_embeds": inputs_embeds}

        if attention_mask is None:
            attention_mask = np.ones((inputs_embeds.shape[0], inputs_embeds.shape[1]), dtype=int)
        inputs["attention_mask"] = attention_mask
        # Run inference
        last_hidden_state = torch.from_numpy(self.request.infer(inputs, share_inputs=True, share_outputs=True)["last_hidden_state"]).to(self.device)

        return BaseModelOutput(last_hidden_state=last_hidden_state)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class OVDecoder:
    """
    Decoder model for OpenVINO inference.

    Arguments:
        request (`openvino.runtime.ie_api.InferRequest`):
            The OpenVINO inference request associated to the decoder.
        device (`torch.device`):
            The device type used by this process.
    """

    def __init__(self, model_path, parent_model, device, ov_config):
        self.model = core.read_model(model_path)
        self._device = device
        self.parent_model = parent_model
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.input_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.inputs}
        self.key_value_input_names = [key for key in self.input_names if "key_value" in key]
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.output_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.outputs}
        self.key_value_output_names = [key for key in self.output_names if "key_values" in key or "present" in key]

        if len(self.key_value_input_names) > 0:
            self.use_past = True
            self.num_pkv = 2
        else:
            self.use_past = False
            self.num_pkv = 4

        compiled_model = core.compile_model(self.model, self._device, ov_config)
        self.request = compiled_model.create_infer_request()

    @property
    def device(self) -> torch.device:
        return self.parent_model.device

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return torch.float32

    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:
        # Model inputs
        inputs = {}

        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = tuple(past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer)

            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        inputs["decoder_input_ids"] = input_ids  # self.parent_model.encoder.input_embedding(input_ids)[0]
        inputs["encoder_hidden_states"] = encoder_hidden_states

        # Add the encoder_attention_mask inputs when needed
        inputs["encoder_attention_mask"] = encoder_attention_mask

        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(self.device)

        # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
        # self-attention layer and 2 to the cross-attention layer)
        out_past_key_values = tuple(self.request.get_tensor(key).data for key in list(self.key_value_output_names))
        # print(len(out_past_key_values))
        # Tuple of tuple of length `n_layers`, with each tuple of length equal to:
        # * 4 for the decoder without cache (k/v of self-attention + k/v of cross-attention)
        # * 2 for the decoder with cache (k/v of self-attention as cross-attention cache is constant)
        if self.use_past is False:
            out_past_key_values = tuple(out_past_key_values[i : i + 4] for i in range(0, len(out_past_key_values), 4))
        else:
            # grab the cross attention key/values from the inputs
            out_past_key_values = tuple(
                out_past_key_values[i : i + self.num_pkv] + past_key_values[2 * i + 2 : 2 * i + 2 + self.num_pkv]
                for i in range(0, len(out_past_key_values), self.num_pkv)
            )
        return Seq2SeqLMOutput(logits=logits, past_key_values=out_past_key_values)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class OVFlorence2Model:
    def __init__(self, model_dir, device, ov_config=None) -> None:
        model_dir = Path(model_dir)
        ov_config = ov_config or {}
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.image_embedding = core.compile_model(model_dir / IMAGE_EMBEDDING_NAME, device, ov_config)
        self.text_embedding = core.compile_model(model_dir / TEXT_EMBEDING_NAME, device, ov_config)
        self.language_model = OVFlorence2LangModel(model_dir, self.config.text_config, device, ov_config)

    def generate(self, input_ids, inputs_embeds=None, pixel_values=None, **kwargs):
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings(input_ids)
            # 2. Merge text and images
            if pixel_values is not None:
                image_features = self._encode_image(pixel_values)
                inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(image_features, inputs_embeds)
        return self.language_model.generate(input_ids=None, inputs_embeds=torch.from_numpy(inputs_embeds), **kwargs)

    def get_input_embeddings(self, input_ids):
        return self.language_model.get_input_embeddings(input_ids)

    def _encode_image(self, pixel_values):
        return self.image_embedding(pixel_values)[0]

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds):
        batch_size, image_token_length = image_features.shape[:-1]
        image_attention_mask = np.ones((batch_size, image_token_length))

        # task_prefix_embeds: [batch_size, padded_context_length, hidden_size]
        # task_prefix_attention_mask: [batch_size, context_length]
        if inputs_embeds is None:
            return image_features, image_attention_mask

        task_prefix_embeds = inputs_embeds
        task_prefix_attention_mask = np.ones((batch_size, task_prefix_embeds.shape[1]))

        if len(task_prefix_attention_mask.shape) == 3:
            task_prefix_attention_mask = task_prefix_attention_mask[:, 0]

        # concat [image embeds, task prefix embeds]
        inputs_embeds = np.concatenate([image_features, task_prefix_embeds], axis=1)
        attention_mask = np.concatenate([image_attention_mask, task_prefix_attention_mask], axis=1)

        return inputs_embeds, attention_mask


class OVFlorence2LangModel(GenerationMixin):
    def __init__(self, model_dir, config, device, ov_config):
        self.config = config
        self.generation_config = GenerationConfig.from_model_config(config)
        self.encoder = OVEncoder(model_dir, self, device, ov_config)
        self.decoder = OVDecoder(model_dir / DECODER_NAME, self, device, ov_config)
        self.decoder_with_past = OVDecoder(model_dir / DECODER_WITH_PAST_NAME, self, device, ov_config)
        self.base_model_prefix = "openvino_model"
        self.main_input_name = "input_ids"
        self._supports_cache_class = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def can_generate(self):
        return True

    def get_input_embeddings(self, input_ids):
        return self.encoder.input_embedding(input_ids)[0]

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)

        # Decode
        if past_key_values is None or self.decoder_with_past is None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                # ecoder_attention_mask=decoder_attention_mask,
            )
            logits = decoder_outputs[0]
        else:
            decoder_outputs = self.decoder_with_past(
                input_ids=decoder_input_ids,  # Cut decoder_input_ids if past is used
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                # decoder_attention_mask=decoder_attention_mask,
            )
            logits = decoder_outputs[0]

        return Seq2SeqLMOutput(logits=logits, past_key_values=decoder_outputs.past_key_values)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # Cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (tuple(np.take(past_state, beam_idx, 0) for past_state in layer_past[:2]) + layer_past[2:],)
        return reordered_past

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return torch.float32

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
