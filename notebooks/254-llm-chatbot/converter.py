from functools import wraps
import torch
import openvino as ov
from pathlib import Path
from typing import Tuple


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


def convert_chatglm2(pt_model: torch.nn.Module, model_path: Path):
    """
    ChatGLM model conversion function

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


converters = {
    "mpt": convert_mpt,
    "qwen": convert_qwen,
    "chatglm2": convert_chatglm2,
}
