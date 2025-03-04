import shutil
import openvino as ov
from transformers import GenerationMixin, AutoConfig, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import nncf
from pathlib import Path
import torch
import types
from typing import Optional, Tuple, List
import openvino.opset13 as opset13
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
import numpy as np
from einops import rearrange, repeat
import gc


model_ids = [
    "deepseek-ai/deepseek-vl2-tiny",
    "deepseek-ai/deepseek-vl2-small",
]

model_id = model_ids[0]

model_path = Path(model_id.split("/")[-1])
VISION_EMBEDDINGS_PATH = "openvino_vision_embeddings_model.xml"
TEXT_EMBEDDINGS_PATH = "openvino_text_embeddings_model.xml"
LANGUAGE_MODEL_PATH = "openvino_language_model.xml"


def prepare_model_repo():
    from cmd_helper import clone_repo

    repo_path = clone_repo("https://github.com/deepseek-ai/DeepSeek-VL2.git")

    modeling_helper = Path("modeling_helper.py")
    dst_modeling_helper = repo_path / "deepseek_vl2/models/modeling_helper.py"
    if not dst_modeling_helper.exists():
        shutil.copy(modeling_helper, dst_modeling_helper)

    deepseek_modeling_path = repo_path / "deepseek_vl2/models/modeling_deepseek.py"
    orig_deepseek_modeling_path = repo_path / "deepseek_vl2/models/orig_modeling_deepseek.py"
    if not orig_deepseek_modeling_path.exists():
        shutil.move(deepseek_modeling_path, orig_deepseek_modeling_path)
        with orig_deepseek_modeling_path.open("r", encoding="utf-8") as f:
            text = f.read()
            updated_text = text.replace("from transformers.models.llama.modeling_llama", "from .modeling_helper")
            with deepseek_modeling_path.open("w", encoding="utf-8") as out_f:
                out_f.write(updated_text)
    orig_siglip_path = repo_path / "deepseek_vl2/models/orig_siglip_vit.py"
    siglip_path = repo_path / "deepseek_vl2/models/siglip_vit.py"

    if not orig_siglip_path.exists():
        shutil.move(siglip_path, orig_siglip_path)
        with orig_siglip_path.open("r", encoding="utf-8") as f:
            text = f.read()
            updated_text = text.replace("from xformers.ops", "from .modeling_helper")
            with siglip_path.open("w", encoding="utf-8") as out_f:
                out_f.write(updated_text)


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


def deepseek_v2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # modified from https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py#L806
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)

        b, h, s, d = q.shape
        q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    if output_attentions:
        return self._orig_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    bsz, q_len, _ = hidden_states.shape

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)

    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    kv_seq_len = value_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    # Difference with original code, k_pe.new_empty create constant tensor in torchscript
    query_states = torch.concat([q_nope, q_pe], dim=-1)
    # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    # query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
    key_states = torch.concat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)
    # key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    # key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")
    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal=self.is_causal and attention_mask is None and q_len > 1,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def deepseek_moe_infer(self, x, topk_ids, topk_weight):
    cnts = torch.zeros((topk_ids.shape[0], len(self.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0).to(torch.long)
    idxs = torch.argsort(topk_ids.view(-1))
    sorted_tokens = x[idxs // topk_ids.shape[1]]

    outputs = []
    start_idx = torch.tensor(0, dtype=torch.long)
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        # difference with original: removed skiping expert if empty num_tokens
        expert_id = i + self.ep_rank * self.experts_per_rank
        expert = self.experts[expert_id]
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = expert(tokens_for_this_expert)
        outputs.append(expert_out)
        start_idx = end_idx

    # difference with original: removed usage torch.new_empty if outputs empty
    outs = torch.cat(outputs, dim=0)

    new_x = torch.zeros_like(outs)
    new_x[idxs] = outs
    final_out = new_x.view(*topk_ids.shape, -1).to(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).to(new_x.dtype)
    return final_out


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def convert_deepseek_vl(model_id=model_id, model_path=None, quantization_config=None):
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

    if model_path is None:
        model_path = Path(model_id.split("/")[-1])

    if all((model_path / model_name).exists() for model_name in [VISION_EMBEDDINGS_PATH, TEXT_EMBEDDINGS_PATH, LANGUAGE_MODEL_PATH]):
        print(f"✅ {model_id} model already converted. You can find results in {model_path}")
        return model_path
    print(f"⌛ {model_id} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")
    pt_model = DeepseekVLV2ForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    pt_model.eval()
    processor = DeepseekVLV2Processor.from_pretrained(model_id)
    processor.save_pretrained(model_path)
    config = pt_model.config
    if pt_model.tile_tag == "2D":
        config.image_newline = pt_model.image_newline.tolist()
        config.view_seperator = pt_model.view_seperator.tolist()
    else:
        config.tile_indicators = pt_model.tile_indicators.tolist()
    config.save_pretrained(model_path)
    __make_16bit_traceable(pt_model)
    print("✅ Original model successfully loaded")
    if not (model_path / TEXT_EMBEDDINGS_PATH).exists():
        print("⌛ Convert Input embedding model")
        ov_model = ov.convert_model(pt_model.language.get_input_embeddings(), example_input=torch.ones([2, 2], dtype=torch.long))
        ov.save_model(ov_model, model_path / TEXT_EMBEDDINGS_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Input embedding model successfully converted")
    if not (model_path / VISION_EMBEDDINGS_PATH).exists():
        print("⌛ Convert Image embedding model")

        def vision_forward(self, pixel_values):
            images_feature = self.vision(pixel_values)
            # [batch_all_tiles, hw, D]
            images_embeds = self.projector(images_feature)
            return images_embeds

        pt_model._orig_forward = pt_model.forward
        pt_model.forward = types.MethodType(vision_forward, pt_model)

        ov_model = ov.convert_model(pt_model, example_input=torch.zeros([2, 3, 384, 384]))

        if quantization_config is not None and "vision" in quantization_config:
            nncf.compress_weights(ov_model, **quantization_config["vision"])
        ov.save_model(ov_model, model_path / VISION_EMBEDDINGS_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Image embedding model successfully converted")

    if not (model_path / LANGUAGE_MODEL_PATH).exists():
        lm = pt_model.language
        inputs_embeds = torch.zeros([2, 2, lm.config.hidden_size], dtype=torch.float32)
        attention_mask = torch.ones([2, 4], dtype=torch.long)
        position_ids = torch.tensor([[2, 3], [2, 3]])
        pkv_input_names = []
        pkv_inputs = []
        pkv_output_names = []
        v_head_dim = lm.config.hidden_size // lm.config.num_attention_heads if not lm.config.use_mla else lm.config.v_head_dim
        k_head_dim = lm.config.qk_nope_head_dim + lm.config.qk_rope_head_dim if lm.config.use_mla else v_head_dim
        num_layers = lm.config.num_hidden_layers
        v_shape = (2, lm.config.num_key_value_heads, 2, v_head_dim)
        k_shape = (2, lm.config.num_key_value_heads, 2, k_head_dim)
        for block in lm.model.layers:
            if lm.config.use_mla:
                block.self_attn.forward = types.MethodType(deepseek_v2_attn_forward, block.self_attn)
            if hasattr(block.mlp, "moe_infer"):
                block.mlp._org_moe_infer = block.mlp.moe_infer
                block.mlp.moe_infer = types.MethodType(deepseek_moe_infer, block.mlp)
        for idx in range(num_layers):
            pkv_inputs.append((torch.randn(k_shape), torch.randn(v_shape)))
            pkv_input_names.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
            pkv_output_names.extend([f"present.{idx}.key", f"present.{idx}.value"])

        model_inputs = ["attention_mask", "position_ids", *pkv_input_names, "inputs_embeds"]
        model_outputs = ["logits", *pkv_output_names]

        lm.config.torchscript = True
        dummy_inputs = {"attention_mask": attention_mask, "position_ids": position_ids, "past_key_values": pkv_inputs, "inputs_embeds": inputs_embeds}
        ts_decoder = TorchScriptPythonDecoder(lm, example_input=dummy_inputs, trace_kwargs={"check_trace": False})

        ov_model = ov.convert_model(ts_decoder, example_input=dummy_inputs)
        for input, input_name in zip(ov_model.inputs, model_inputs):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, model_outputs):
            output.get_tensor().set_names({output_name})
        patch_stateful(ov_model)
        print("✅ Language model successfully converted")
        if quantization_config is not None and "llm" in quantization_config:
            print(f"⌛ Weights compression with {quantization_config['llm']['mode']} mode started")
            ov_model = nncf.compress_weights(ov_model, **quantization_config["llm"])
            print("✅ Weights compression finished")
        else:
            ov_model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
        ov.save_model(ov_model, model_path / LANGUAGE_MODEL_PATH)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
    del pt_model
    gc.collect()
    print(f"✅ {model_id} model conversion finished. You can find results in {model_path}")
    return model_path


class OvModelForCausalLMWithEmb(GenerationMixin):
    def __init__(self, model_dir, device="CPU", config=None, ov_config=None, compile=True) -> None:
        self._supports_cache_class = False
        self.config = AutoConfig.from_pretrained(model_dir).text_config if config is None else config
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False
        self.generation_config = GenerationConfig.from_model_config(self.config)
        model_dir = Path(model_dir)
        self.model = core.read_model(model_dir / LANGUAGE_MODEL_PATH)
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


core = ov.Core()


class OVDeepseekVLV2ForCausalLM(GenerationMixin):
    def __init__(self, model_dir, device="CPU", ov_config=None):
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config

        self.config = DeepseekVLV2Config.from_pretrained(model_dir)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.language_model = OvModelForCausalLMWithEmb(model_dir, device, self.config.language_config, ov_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False
        self._supports_cache_class = False
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self.vision_embeddings = core.compile_model(model_dir / VISION_EMBEDDINGS_PATH, device, ov_config)
        if self.config.tile_tag == "2D":
            self.image_newline = torch.tensor(self.config.image_newline)
            self.view_seperator = torch.tensor(self.config.view_seperator)
        else:
            self.tile_indicators = torch.tensor(self.config.tile_indicators)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.LongTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        **ignore_kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            images (torch.FloatTensor): [b, max_n_images, 3, height, width]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_spatial_crop (torch.LongTensor): [b, max_n_images, 2]

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        if images is None or images_spatial_crop.sum() == 0:
            return torch.from_numpy(self.language_model.embed_tokens(input_ids))

        bs, max_n_images, _ = images_spatial_crop.shape
        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += 1 + num_width_tiles * num_height_tiles

            total_tiles.append(images[idx, : batch_num_tiles[idx]])

        # [batch_all_tiles, 3, height, width]
        total_tiles = torch.cat(total_tiles, dim=0)
        assert total_tiles.shape[0] == sum(batch_num_tiles)
        if total_tiles.shape[0] == 0:
            return torch.from_numpy(self.language_model.embed_tokens(input_ids))

        # [batch_all_tiles, vit_seq_len, c])
        images_embeds = torch.from_numpy(self.vision_embeddings(total_tiles)[0])

        _, hw, n_dim = images_embeds.shape
        h = w = int(hw**0.5)

        # put image tokens into the input_embeds, [b, T, D]
        input_embeds = torch.from_numpy(self.language_model.embed_tokens(input_ids))

        # 根据self.tile_tag & self.global_view_pos填充image token sequence
        tile_index = 0
        for idx in range(images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(images_spatial_crop.shape[1]):

                # extra global & local features
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles

                # [hw, D]
                global_features = images_embeds[tile_index]

                # [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embeds[tile_index + 1 : tile_index + 1 + num_tiles_in_image]

                tile_index += num_tiles_in_image + 1

                # format global and local features
                if self.config.tile_tag == "2D":

                    # ----------------- global view add newline -----------------
                    # [hw, D] -> [h, w, D]
                    global_features = global_features.view(h, w, n_dim)
                    # [D]     -> [h, 1, D]
                    new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
                    # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                    global_features = torch.cat([global_features, new_lines_in_global], dim=1)
                    # [h, w + 1, D] -> [h * (w + 1), D]
                    global_features = global_features.view(-1, n_dim)

                    # ----------------- local view add newline -----------------
                    # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
                    local_features = rearrange(local_features, "(th tw) (h w) d -> (th h) (tw w) d", th=num_height_tiles, tw=num_width_tiles, h=h, w=w)

                    # [D] -> [num_height_tiles * h, 1, D]
                    new_lines_in_local = repeat(self.image_newline, "d -> (th h) 1 d", th=num_height_tiles, h=h)

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    local_features = torch.cat([local_features, new_lines_in_local], dim=1)

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                    local_features = local_features.view(-1, n_dim)

                    # ----------------- merge global and local tiles -----------------
                    if self.config.global_view_pos == "head":
                        global_local_features = torch.cat([global_features, self.view_seperator[None, :], local_features], dim=0)
                    else:
                        global_local_features = torch.cat([local_features, self.view_seperator[None, :], global_features], dim=0)

                else:
                    # abandoned，实际上不会走这个逻辑
                    global_features = torch.cat([self.tile_indicators[0:1], global_features], dim=0)
                    local_features = torch.cat([self.tile_indicators[1 : num_tiles_in_image + 1].unsqueeze(1), local_features], dim=1)
                    local_features = rearrange(local_features, "crop_num hw d -> (crop_num hw) d")

                    if self.config.global_view_pos == "head":
                        global_local_features = torch.cat([global_features, local_features], dim=0)
                    else:
                        global_local_features = torch.cat([local_features, global_features], dim=0)

                images_in_this_batch.append(global_local_features)

            if len(images_in_this_batch) > 0:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                input_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1), images_in_this_batch)

        return input_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.LongTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):

        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )

        outputs = self.language_model.forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.LongTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        attention_mask=None,
        cache_position=None,
        pixel_values=None,
        image_sizes=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model
        cache_position = model_inputs["cache_position"]
        if cache_position[0] == 0:
            model_inputs["images"] = images
            model_inputs["images_seq_mask"] = images_seq_mask
            model_inputs["images_spatial_crop"] = images_spatial_crop

        return model_inputs

    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        return self.language_model._reorder_cache(past_key_values, beam_idx)

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""

        return True
