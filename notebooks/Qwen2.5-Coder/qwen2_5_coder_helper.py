import gc
import os
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np

# Fix DLL search path on Windows to avoid loading wrong torch/openvino DLLs
# from other conda environments (e.g. project env with torch 2.12)
_ov_env = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages')
for _d in ['torch/lib', 'openvino/libs']:
    _p = os.path.join(_ov_env, _d)
    if os.path.isdir(_p) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(_p)

# torch MUST be imported before openvino to avoid DLL load failures
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.generation import GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache

import openvino as ov

try:
    from openvino import opset13
except ImportError:
    from openvino.runtime import opset13

core = ov.Core()

LANGUAGE_MODEL_NAME = "openvino_language_model.xml"
EMBEDDING_MODEL_NAME = "openvino_embedding.xml"
LM_HEAD_NAME = "openvino_lm_head.xml"


def model_has_state(ov_model: ov.Model):
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: list,
    key_value_input_names: list,
    gather_dim: int,
):
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
    not_kv_inputs: list,
    key_value_input_names: list,
    key_value_output_names: list,
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
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
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def convert_qwen2_5_coder_model(model_id, output_dir, quantization_config=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_model_path = output_dir / LANGUAGE_MODEL_NAME
    embedding_path = output_dir / EMBEDDING_MODEL_NAME
    lm_head_path = output_dir / LM_HEAD_NAME

    if lang_model_path.exists() and embedding_path.exists() and lm_head_path.exists():
        print(f"[OK] {model_id} model already converted. Results in {output_dir}")
        return

    print(f"[...] {model_id} conversion started. Be patient, it may take some time.")
    print("[...] Loading original model")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    del tokenizer
    gc.collect()

    print("[OK] Original model successfully loaded")

    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers

    # === Convert Embedding Model ===
    if not embedding_path.exists():
        print("[...] Converting embedding model")
        embed_tokens = model.model.embed_tokens
        ov_model = ov.convert_model(
            embed_tokens,
            example_input=torch.ones([1, 1], dtype=torch.int64),
        )
        ov.save_model(ov_model, embedding_path)
        del ov_model, embed_tokens
        cleanup_torchscript_cache()
        gc.collect()
        print("[OK] Embedding model successfully converted")

    # === Convert LM Head Model ===
    if not lm_head_path.exists():
        print("[...] Converting LM head model")
        lm_head = model.lm_head

        class LMHeadWrapper(torch.nn.Module):
            def __init__(self, lm_head):
                super().__init__()
                self.lm_head = lm_head

            def forward(self, hidden_states):
                return self.lm_head(hidden_states)

        wrapper = LMHeadWrapper(lm_head)
        ov_model = ov.convert_model(
            wrapper,
            example_input=torch.randn([1, 1, hidden_size], dtype=torch.float16),
        )
        ov.save_model(ov_model, lm_head_path)
        del ov_model, wrapper, lm_head
        model.lm_head = None
        cleanup_torchscript_cache()
        gc.collect()
        print("[OK] LM head model successfully converted")

    # === Convert Language Model (main transformer) ===
    if not lang_model_path.exists():
        print("[...] Converting language model (this may take a while)")

        lang_model = model.model
        num_pkv = num_layers
        embedding_size = hidden_size

        # Free embedding layer from main model to save memory
        model.model.embed_tokens = None
        gc.collect()

        lang_model._orig_forward = lang_model.forward

        def forward_wrap(
            self,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            outputs = self._orig_forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache if use_cache is not None else True,
                output_attentions=output_attentions if output_attentions is not None else False,
                output_hidden_states=output_hidden_states if output_hidden_states is not None else False,
                return_dict=return_dict if return_dict is not None else True,
            )

            if return_dict and hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                if isinstance(outputs.past_key_values, DynamicCache):
                    pkv = outputs.past_key_values.to_legacy_cache()
                    return type('ModelOutput', (), {
                        'last_hidden_state': outputs.last_hidden_state,
                        'past_key_values': pkv,
                    })()
            return outputs

        lang_model.forward = types.MethodType(forward_wrap, lang_model)

        pkv_shape = (
            2,
            config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads,
            2,
            config.head_dim if hasattr(config, "head_dim") else hidden_size // config.num_attention_heads,
        )

        cache_position = torch.arange(2, 4)
        position_ids = cache_position.view(1, 1, -1).expand(3, 2, -1)

        input_embeds = torch.randn((2, 2, embedding_size), dtype=torch.float16)
        attention_mask = torch.ones([2, 4], dtype=torch.long)

        input_names = ["attention_mask", "position_ids"]
        output_names = ["hidden_states"]

        past_key_values = []
        for i in range(num_pkv):
            kv = [torch.randn(pkv_shape, dtype=torch.float16) for _ in range(2)]
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

        head_dim = pkv_shape[-1]
        input_shapes = [
            ov.PartialShape([-1, -1]),
            ov.PartialShape([3, -1, -1]),
        ]
        input_shapes += (
            [
                ov.PartialShape(
                    [
                        -1,
                        config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads,
                        -1,
                        head_dim,
                    ]
                )
            ]
            * 2
            * num_pkv
        )
        input_shapes += [ov.PartialShape([-1, -1, embedding_size])]

        ov_model = ov.convert_model(lang_model, example_input=example_input, input=input_shapes)

        for input, input_name in zip(ov_model.inputs, input_names):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})

        patch_stateful(ov_model, 2)
        print("[OK] Language model successfully converted")

        if quantization_config is not None:
            import nncf
            print(f"[...] Weights compression with {quantization_config['mode']} started")
            ov_model = nncf.compress_weights(ov_model, **quantization_config)
            print("[OK] Weights compression finished")

        ov.save_model(ov_model, lang_model_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print(f"[OK] Language model conversion finished. Results in {output_dir}")

    del model
    gc.collect()


class OVQwen2_5CoderForCausalLM(GenerationMixin):
    _is_stateful = False

    def __init__(self, model_dir: Path, device: str, config):
        self.model_dir = Path(model_dir)
        self.config = config
        self.device = torch.device("cpu")
        self.dtype = torch.float16

        self.model = core.read_model(model_dir / LANGUAGE_MODEL_NAME)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        compiled_model = core.compile_model(self.model, device)
        self.request = compiled_model.create_infer_request()

        self.embed_tokens = core.compile_model(model_dir / EMBEDDING_MODEL_NAME, device)
        self.lm_head_model = core.compile_model(model_dir / LM_HEAD_NAME, device)

        self._embedding_wrapper = self._create_embedding_wrapper()

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
        def embedding_fn(input_ids):
            if isinstance(input_ids, torch.Tensor):
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

    def get_input_embeddings(self):
        return self._embedding_wrapper

    def get_rope_index(self, attention_mask):
        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        return position_ids, mrope_position_deltas

    def can_generate(self):
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
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            pass
        else:
            inputs_embeds = self._embedding_wrapper(input_ids)

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

        if past_key_values is None:
            self.request.reset_state()
            self.next_beam_idx = np.arange(inputs_embeds.shape[0], dtype=int)
            self._past_length = 0

        inputs = {
            "inputs_embeds": inputs_embeds.numpy() if isinstance(inputs_embeds, torch.Tensor) else inputs_embeds,
            "attention_mask": attention_mask.numpy() if isinstance(attention_mask, torch.Tensor) else attention_mask,
            "position_ids": position_ids.numpy() if isinstance(position_ids, torch.Tensor) else position_ids,
        }

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(inputs_embeds.shape[0], dtype=int)

        self.request.start_async(inputs, share_inputs=False)
        self.request.wait()

        hidden_states = torch.from_numpy(self.request.get_tensor("hidden_states").data.copy()).to(self.device)
        logits = torch.from_numpy(self.lm_head_model(hidden_states.numpy())[0]).to(self.device)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=((),),
            hidden_states=None,
            attentions=None,
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, num_new_tokens)
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
