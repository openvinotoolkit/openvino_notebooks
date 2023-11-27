import numpy as np
import openvino as ov
from openvino._pyopenvino import Node
from openvino.runtime.op import Parameter, Constant
from openvino.runtime.opset12 import matmul, convolution
from openvino.runtime.utils.types import get_element_type
from typing import List, Dict, Union, Tuple

ops_to_track_map = {
    'Convolution': convolution,
    'MatMul': matmul,
}


def get_thresholds_per_op():
    return {
        'Convolution': (0.1, 0.003, 0.00),
        'MatMul': (0.1, 0.04, 0.03),
    }


def rt_info_name_to_keep_orig_precision():
    return 'precise_0'


def partially_upcast_nodes_to_fp32(orig_model: ov.Model, example_input: Union[List, Dict],
                                   thresholds_per_op: Dict[str, Tuple] = None) -> ov.Model:
    model = orig_model.clone()  # todo: check if need to clone orig_models
    nodes_to_track, outs_to_track, _ = insert_results_for_tracked_ops(model)
    fp16_full_net_infer_values = infer_full_net_in_fp16(nodes_to_track, model, example_input)
    fp16_infer_values = infer_nodes_in_fp16(nodes_to_track, fp16_full_net_infer_values)
    fp32_infer_values = infer_nodes_in_fp32(nodes_to_track, fp16_full_net_infer_values)
    del model
    new_model = orig_model.clone()
    mark_nodes_to_upcast_to_fp32(new_model, nodes_to_track, fp16_infer_values, fp32_infer_values, thresholds_per_op)
    return new_model


def insert_results_for_tracked_ops(model) -> (List, List, List):
    # additional outputs to track inputs and output values of operations of interest
    nodes_to_track = []
    outputs = []
    for i, op in enumerate(model.get_ordered_ops()):
        if op.get_type_name() not in ops_to_track_map.keys():
            continue
        if any(map(lambda input: input.get_node().get_type_name() == 'Result', op.output(0).get_target_inputs())):
            continue
        outputs.append(op.output(0))
        nodes_to_track.append(op)

        node_0 = op.input_value(0).get_node()
        node_1 = op.input_value(1).get_node()
        for node in [node_0, node_1]:
            if node.get_type_name() != 'Constant' and not is_decompression_convert(node):  # for Consts we can take inputs from ov::Model
                outputs.append(node.output(0))

    orig_outputs = model.outputs.copy()
    model.add_outputs(outputs)
    return nodes_to_track, outputs, orig_outputs


def get_const_value_from_ovmodel(node: Union[Constant, Node]) -> np.ndarray:
    if node.get_type_name() == 'Constant':
        assert node.get_element_type() == ov.Type.f32
        return node.get_data()
    elif is_decompression_convert(node):
        # if model is compressed and constant values flow through decompression convert
        const_node = node.input_value(0).get_node()
        assert const_node.get_type_name() == 'Constant'
        assert const_node.get_element_type().is_real()
        return np.array(node.input_value(0).get_node().get_data(), dtype=np.float32)
    else:
        raise Exception(
            f'Cannot get const values from ov.Model for node {node.get_friendly_name()} with type {node.get_type_name()}')


def is_decompression_convert(node: Node) -> bool:
    if node.get_type_name() != 'Convert':
        return False
    if len(node.get_rt_info()['is_decompression_0'].aslist()) > 0:
        return True
    return False


def infer_full_net_in_fp16(nodes_to_track: List[Node], orig_model: ov.Model, example_inputs: List) -> List[Tuple]:
    core = ov.Core()
    assert 'GPU' in core.available_devices
    exec_net = core.compile_model(orig_model, 'GPU', config={"INFERENCE_PRECISION_HINT": "f16"})
    request = exec_net.create_infer_request()
    results = request.infer(example_inputs)

    results_map = {}
    for key, val in results.items():
        for input_val in key.node.input_values():
            node_name = input_val.get_node().get_friendly_name()
            if input_val.get_node().get_type_name() == 'Constant' or is_decompression_convert(input_val.get_node()):
                results_map[node_name] = get_const_value_from_ovmodel(input_val.get_node())
            else:
                results_map[node_name] = val

    node_data_values = []  # each item contains a tuple with node output values and all input values
    for node in nodes_to_track:
        res_item = [results_map[node.get_friendly_name()]]
        for input_val in node.input_values():
            if input_val.get_node().get_type_name() == 'Constant' or is_decompression_convert(input_val.get_node()):
                res_item.append(get_const_value_from_ovmodel(input_val.get_node()))
            else:
                res_item.append(results_map[input_val.get_node().get_friendly_name()])
        node_data_values.append(tuple(res_item))
    del request, exec_net
    return node_data_values


def infer_nodes_in_fp32(nodes_to_track: List[Node], node_data_values: List[Tuple]) -> List:
    results = []
    for node, value in zip(nodes_to_track, node_data_values):
        results.append(infer_tracked_op_on_gpu(node, value[1:]))
    return results


def infer_nodes_in_fp16(nodes_to_track: List[Node], node_data_values: List[Tuple]) -> List:
    results = []
    for node, value in zip(nodes_to_track, node_data_values):
        results.append(infer_tracked_op_on_gpu(node, value[1:], precision='f16'))
    return results


def infer_tracked_op_on_gpu(op: Node, input_vals: Tuple, precision='f32') -> np.ndarray:
    parameters = []
    for input_val in input_vals:
        parameters.append(Parameter(get_element_type(input_val.dtype), ov.PartialShape(input_val.shape)))

    if op.get_type_name() not in ops_to_track_map.keys():
        # todo: implement for other ops
        raise NotImplementedError(f"inference track for operations {op.get_type_name()} are not implemented yet")

    new_op = ops_to_track_map[op.get_type_name()](*parameters, **op.get_attributes())
    ov_model = ov.Model([new_op], parameters)

    exec_net = ov.Core().compile_model(ov_model, 'GPU', config={"INFERENCE_PRECISION_HINT": precision})
    request = exec_net.create_infer_request()
    result = request.infer(input_vals)
    assert len(result) == 1
    del request, exec_net, ov_model
    return result[0]


def is_model_partially_upcasted(model) -> bool:
    for node in model.get_ordered_ops():
        if node.get_type_name() not in ops_to_track_map.keys():
            continue
        if rt_info_name_to_keep_orig_precision() in node.get_rt_info().keys():
            return True
    return False


def mark_nodes_to_upcast_to_fp32(model: ov.Model, nodes: List[Node], fp16_infer_vals: List, fp32_infer_vals: List,
                                 thresholds: None) -> None:
    nodes_with_errors = []
    for node, fp16_val, fp32_val in zip(nodes, fp16_infer_vals, fp32_infer_vals):
        if compare_tensors(node, fp16_val, fp32_val, thresholds):
            nodes_with_errors.append(node.get_friendly_name())

    for node in model.get_ordered_ops():
        if node.get_friendly_name() in nodes_with_errors:
            node.get_rt_info()[rt_info_name_to_keep_orig_precision()] = ''


def compare_tensors(node: Node, a: np.ndarray, b: np.ndarray, new_thresholds_per_op, silent: bool = True) -> bool:
    """
    If values differ more than a certain metric then function returns True
    """
    assert np.array_equal(a.shape, b.shape), f'Shapes differ {a.shape} and {b.shape}'
    out_size = int(np.prod(a.shape))
    a_, b_ = np.reshape(a, out_size), np.reshape(b, out_size)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rel_error = np.abs(2 * (a_ - b_) / (np.abs(a_) + abs(b_)))

    mean_rel_error = np.mean(rel_error)
    thresholds_map = get_thresholds_per_op()
    if new_thresholds_per_op is not None:
        thresholds_map.update(new_thresholds_per_op)
    thresholds = thresholds_map[node.get_type_name()]
    rel_threshold = thresholds[0]
    rel_threshold_ratio = thresholds[1]
    rel_tol = thresholds[2]

    rel_diff_ratio = np.size(np.where(rel_error >= rel_threshold)) / out_size
    if mean_rel_error < rel_tol:
        return False
    if rel_diff_ratio > rel_threshold_ratio:
        if not silent:
            print(f'upcasted node {node.get_friendly_name()} with 0.1 rel2_diff_ratio {rel_diff_ratio} and mean_rel_error {mean_rel_error}')
        return True
