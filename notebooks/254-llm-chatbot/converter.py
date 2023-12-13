from functools import wraps
import torch
import openvino as ov
from pathlib import Path
from typing import Tuple, Optional
import types
from transformers.modeling_outputs import BaseModelOutputWithPast


def flattenize_inputs(inputs):
    """
    Helper function for making nested inputs flattens
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def convert_mpt(pt_model: torch.nn.Module, model_path: Path):
    """
    MPT model conversion function

    Params:
      pt_model: PyTorch model
      model_path: path for saving model
    Returns:
      None
    """
    ov_out_path = Path(model_path) / "openvino_model.xml"
    pt_model.config.save_pretrained(ov_out_path.parent)
    pt_model.config.use_cache = True
    outs = pt_model(
        input_ids=torch.ones((1, 10), dtype=torch.long),
        attention_mask=torch.ones((1, 10), dtype=torch.long),
    )
    inputs = ["input_ids"]
    outputs = ["logits"]

    dynamic_shapes = {"input_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"}}
    for idx in range(len(outs.past_key_values)):
        inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
        dynamic_shapes[inputs[-1]] = {2: "past_sequence + sequence"}
        dynamic_shapes[inputs[-2]] = {3: "past_sequence + sequence"}
        outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])

    inputs.append("attention_mask")
    dummy_inputs = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "past_key_values": outs.past_key_values,
        "attention_mask": torch.ones((1, 12), dtype=torch.long),
    }
    pt_model.config.torchscript = True
    orig_forward = pt_model.forward

    @wraps(orig_forward)
    def ts_patched_forward(
        input_ids: torch.Tensor,
        past_key_values: Tuple[Tuple[torch.Tensor]],
        attention_mask: torch.Tensor,
    ):
        pkv_list = list(past_key_values)
        outs = orig_forward(
            input_ids=input_ids, past_key_values=pkv_list, attention_mask=attention_mask
        )
        return (outs.logits, tuple(outs.past_key_values))

    pt_model.forward = ts_patched_forward
    ov_model = ov.convert_model(pt_model, example_input=dummy_inputs)
    pt_model.forward = orig_forward
    for inp_name, m_input, input_data in zip(
        inputs, ov_model.inputs, flattenize_inputs(dummy_inputs.values())
    ):
        input_node = m_input.get_node()
        if input_node.element_type == ov.Type.dynamic:
            m_input.get_node().set_element_type(ov.Type.f32)
        shape = list(input_data.shape)
        if inp_name in dynamic_shapes:
            for k in dynamic_shapes[inp_name]:
                shape[k] = -1
        input_node.set_partial_shape(ov.PartialShape(shape))
        m_input.get_tensor().set_names({inp_name})

    for out, out_name in zip(ov_model.outputs, outputs):
        out.get_tensor().set_names({out_name})

    ov_model.validate_nodes_and_infer_types()
    ov.save_model(ov_model, ov_out_path)
    del ov_model
    cleanup_torchscript_cache()
    del pt_model


def _update_qwen_rotary_embedding_cache(model):
    model.transformer.rotary_emb(2048)


def convert_qwen(pt_model: torch.nn.Module, model_path: Path):
    """
    Qwen model conversion function

    Params:
      pt_model: PyTorch model
      model_path: path for saving model
    Returns:
      None
    """
    _update_qwen_rotary_embedding_cache(pt_model)
    ov_out_path = Path(model_path) / "openvino_model.xml"
    pt_model.config.save_pretrained(ov_out_path.parent)
    pt_model.config.use_cache = True
    outs = pt_model(
        input_ids=torch.ones((1, 10), dtype=torch.long),
        attention_mask=torch.ones((1, 10), dtype=torch.long),
    )
    inputs = ["input_ids"]
    outputs = ["logits"]

    dynamic_shapes = {
        "input_ids": {1: "seq_len"},
        "attention_mask": {1: "seq_len"},
        "token_type_ids": {1: "seq_len"},
    }
    for idx in range(len(outs.past_key_values)):
        inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
        dynamic_shapes[inputs[-1]] = {1: "past_sequence + sequence"}
        dynamic_shapes[inputs[-2]] = {1: "past_sequence + sequence"}
        outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])

    inputs += ["attention_mask", "token_type_ids"]
    dummy_inputs = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "past_key_values": outs.past_key_values,
        "attention_mask": torch.ones((1, 12), dtype=torch.long),
        "token_type_ids": torch.ones((1, 2), dtype=torch.long),
    }
    pt_model.config.torchscript = True
    ov_model = ov.convert_model(pt_model, example_input=dummy_inputs)
    for inp_name, m_input, input_data in zip(
        inputs, ov_model.inputs, flattenize_inputs(dummy_inputs.values())
    ):
        input_node = m_input.get_node()
        if input_node.element_type == ov.Type.dynamic:
            m_input.get_node().set_element_type(ov.Type.f32)
        shape = list(input_data.shape)
        if inp_name in dynamic_shapes:
            for k in dynamic_shapes[inp_name]:
                shape[k] = -1
        input_node.set_partial_shape(ov.PartialShape(shape))
        m_input.get_tensor().set_names({inp_name})

    for out, out_name in zip(ov_model.outputs, outputs):
        out.get_tensor().set_names({out_name})

    ov_model.validate_nodes_and_infer_types()
    ov.save_model(ov_model, ov_out_path)
    del ov_model
    cleanup_torchscript_cache()
    del pt_model


@torch.jit.script_if_tracing
def _chatglm2_get_context_layer(query_layer: torch.Tensor, key_layer: torch.Tensor, value_layer: torch.Tensor):
    mask = torch.zeros((query_layer.shape[-2], key_layer.shape[-2]), dtype=query_layer.dtype)
    if query_layer.shape[2] == key_layer.shape[2]:
        tmp_mask = torch.ones((query_layer.shape[-2], key_layer.shape[-2]), dtype=torch.bool).triu(diagonal=1)
        mask.masked_fill_(tmp_mask, float("-inf"))

    context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=mask)
    return context_layer


def _core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
    query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
    if attention_mask is None:
        context_layer = _chatglm2_get_context_layer(query_layer, key_layer, value_layer)
    else:
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attention_mask
        )
    context_layer = context_layer.permute(2, 0, 1, 3)
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.reshape(*new_context_layer_shape)

    return context_layer


@torch.jit.script_if_tracing
def _get_chatglm_attention_mask(input_ids, past_key):
    mask = torch.zeros((input_ids.shape[1], past_key.shape[0] + input_ids.shape[1]), dtype=past_key.dtype)
    if past_key.shape[0] == 0:
        tmp_mask = torch.ones((input_ids.shape[1], past_key.shape[0] + input_ids.shape[1]), dtype=torch.bool).triu(diagonal=1)
        mask.masked_fill_(tmp_mask, float("-inf"))
    return mask


def _chatglm_transformer_forward(
        self,
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
):
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_size, seq_length = input_ids.shape

    if inputs_embeds is None:
        inputs_embeds = self.embedding(input_ids)

    if self.pre_seq_len is not None:
        if past_key_values is None:
            past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                              dtype=inputs_embeds.dtype)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)), attention_mask], dim=-1)

    if full_attention_mask is None:
        if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
            full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        elif past_key_values is not None:
            full_attention_mask = torch.ones(batch_size, seq_length, seq_length,
                                             device=input_ids.device,
                                             dtype=torch.float) * float("-inf")
            full_attention_mask.triu_(diagonal=1)
            past_length = 0
            if past_key_values:
                past_length = past_key_values[0][0].shape[0]
            if past_length:
                full_attention_mask = torch.cat((torch.zeros(batch_size, seq_length, past_length,
                                                             device=input_ids.device), full_attention_mask), dim=-1)
            full_attention_mask.unsqueeze_(1)

    # Rotary positional embeddings
    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    if position_ids is not None:
        rotary_pos_emb = rotary_pos_emb[position_ids]
    else:
        rotary_pos_emb = rotary_pos_emb[None, :seq_length]
    rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

    # Run encoder.
    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
        kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
    )

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def _patch_chatglm_forward(model: "PreTrainedModel"):
    model.transformer.forward = types.MethodType(_chatglm_transformer_forward, model.transformer)
    for block in model.transformer.encoder.layers:
        block.self_attention.core_attention.forward = types.MethodType(
            _core_attention_forward, block.self_attention.core_attention
        )


def convert_chatglm(pt_model: torch.nn.Module, model_path: Path):
    """
    ChatGLM model conversion function

    Params:
      pt_model: PyTorch model
      model_path: path for saving model
    Returns:
      None
    """
    _patch_chatglm_forward(pt_model)
    ov_out_path = Path(model_path) / "openvino_model.xml"
    pt_model.config.save_pretrained(ov_out_path.parent)
    pt_model.config.use_cache = True
    outs = pt_model(
        input_ids=torch.ones((1, 10), dtype=torch.long),
        position_ids=torch.arange(0, 10, dtype=torch.long),
    )
    inputs = ["input_ids"]
    outputs = ["logits"]

    dynamic_shapes = {
        "input_ids": {1: "seq_len"},
        "position_ids": {1: "seq_len"},
        "attention_mask": {1: "seq_len"},
    }
    inputs += ["position_ids", "attention_mask"]
    for idx in range(len(outs.past_key_values)):
        inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
        dynamic_shapes[inputs[-1]] = {0: "past_sequence + sequence"}
        dynamic_shapes[inputs[-2]] = {0: "past_sequence + sequence"}
        outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])

    dummy_inputs = {
        "input_ids": torch.ones((1, 1), dtype=torch.long),
        "position_ids": torch.tensor([[10]], dtype=torch.long),
        "attention_mask": torch.ones((1, 11), dtype=torch.long),
        "past_key_values": outs.past_key_values,
    }
    pt_model.config.torchscript = True
    ov_model = ov.convert_model(pt_model, example_input=dummy_inputs)
    for inp_name, m_input, input_data in zip(
        inputs, ov_model.inputs, flattenize_inputs(dummy_inputs.values())
    ):
        input_node = m_input.get_node()
        if input_node.element_type == ov.Type.dynamic:
            m_input.get_node().set_element_type(ov.Type.f32)
        shape = list(input_data.shape)
        if inp_name in dynamic_shapes:
            for k in dynamic_shapes[inp_name]:
                shape[k] = -1
        input_node.set_partial_shape(ov.PartialShape(shape))
        m_input.get_tensor().set_names({inp_name})

    for out, out_name in zip(ov_model.outputs, outputs):
        out.get_tensor().set_names({out_name})

    ov_model.validate_nodes_and_infer_types()
    ov.save_model(ov_model, ov_out_path)
    del ov_model
    cleanup_torchscript_cache()
    del pt_model

def convert_mpnet(pt_model: torch.nn.Module, model_path: Path):
    ov_out_path = Path(model_path) / "openvino_model.xml"
    dummy_inputs = {"input_ids": torch.ones((1, 10), dtype=torch.long), "attention_mask": torch.ones(
    (1, 10), dtype=torch.long)}
    ov_model = ov.convert_model(pt_model, example_input=dummy_inputs)
    ov.save_model(ov_model, ov_out_path)
    
def convert_bert(pt_model: torch.nn.Module, model_path: Path):
    ov_out_path = Path(model_path) / "openvino_model.xml"
    dummy_inputs = {"input_ids": torch.ones((1, 10), dtype=torch.long), "attention_mask": torch.ones(
    (1, 10), dtype=torch.long), "token_type_ids": torch.zeros((1, 10), dtype=torch.long)}
    ov_model = ov.convert_model(pt_model, example_input=dummy_inputs)
    ov.save_model(ov_model, ov_out_path)

    
converters = {
    # LLM models
    "mpt": convert_mpt,
    "qwen": convert_qwen,
    "chatglm3": convert_chatglm,
    # embedding models
    "all-mpnet-base-v2": convert_mpnet,
    "text2vec-large-chinese": convert_bert,
}
