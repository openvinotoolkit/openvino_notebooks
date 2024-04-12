from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Union

import numpy as np
from openvino.runtime import Model, Node
from openvino.runtime.op import Parameter, Constant
import openvino.runtime.opset12 as opset
from openvino.runtime.utils.types import get_element_type

import openvino as ov
from tqdm.auto import tqdm

OPERATION_TYPE_MAP = {"MatMul": opset.matmul, "Convolution": opset.convolution}

ORIGINAL_PRECISION_RT_INFO_NAME = "precise_0"


@dataclass
class TrackedNodeInfo:
    """
    Data associated with a node tracked for upcasting
    """

    node: Node  # Target node to track
    snr: float = None  # SNR of the target node
    input_nodes: List[Node] = None  # Input nodes of the target node
    result_node: Node = None  # Result node of the target node
    input_result_nodes: Dict[Node, Node] = (
        None  # Result nodes of non-const inputs of the target node
    )
    node_value_full_precision: np.ndarray = None  # Result of the node in full precision
    node_value_half_precision: np.ndarray = None  # Result of the node in half precision
    input_values_full_precision: np.ndarray = (
        None  # Results of the target node inputs in full precision
    )


def partially_upcast_nodes_to_fp32(
    orig_model: Model,
    example_input: Union[List, Dict],
    half_type: str = "f16",
    batch_size: int = 50,
    operation_types: List[str] = None,
    upcast_ratio: float = 0.1,
    verbose: bool = False,
) -> Model:
    """
    Transform a model to upcast some nodes to be executed in full precision instead of half precision. These nodes are
    marked with runtime info flag.
    Nodes are selected based on Signal-to-Noise Ratio (SNR) metric: upcast_ratio fraction of tracked nodes with the
    lowest SNR are marked for full precision execution.

    Note: Input model should have fp16 weights (i.e. saved with compress_to_fp16=True) in order to conserve
    calibration memory.

    :param orig_model: Model to process
    :param example_input: Example input for model inference
    :param half_type: Either "f16" or "bf16"
    :param batch_size: Number of nodes to process together during a single model inference. The lower the value is,
        the less memory footprint is, but the larger is the processing time. The value of -1 is used to disable
        batching.
    :param operation_types: Types of operations to consider. If None, MatMuls and Convolutions are considered.
    :param upcast_ratio: Fraction of nodes to upcast (with the lowest SNR). 0 - do not upcast anything, 1 - upcast every
        operation of the given types.
    :param verbose: If True, prints progress output.
    :return: Upcasted OV model with some nodes marked for full precision execution.
    """
    if half_type not in ("f16", "bf16"):
        raise ValueError(f"Half type must be either 'f16' or 'bf16'. Got {half_type}.")
    if half_type == "bf16":
        print(
            "Warning! Calibration currently does not provide any improvement for bf16 type."
        )
    if operation_types is None:
        operation_types = ["MatMul", "Convolution"]
    for op_type in operation_types:
        if op_type not in OPERATION_TYPE_MAP:
            raise ValueError(
                f"Operation type must be one of the following {list(OPERATION_TYPE_MAP.keys())}. "
                f"Got {op_type}."
            )
    if verbose:
        print(f"The following operation types will be considered: {operation_types}")

    device = "GPU" if half_type == "f16" else "CPU"

    nodes_to_track_names = get_nodes_to_track(orig_model, operation_types)
    if len(nodes_to_track_names) == 0:
        if verbose:
            print("Warning. Not found any operations of the given type(s).")
        return orig_model.clone()

    node_names_and_snrs = []
    batch_size = (
        len(nodes_to_track_names)
        if batch_size == -1 or batch_size > len(nodes_to_track_names)
        else batch_size
    )
    if verbose:
        print("Started upcasting")
    for i in tqdm(
        range(0, len(nodes_to_track_names), batch_size),
        desc="Processing batches",
        disable=not verbose,
    ):
        if upcast_ratio == 0.0 or upcast_ratio == 1.0:
            continue
        model = orig_model.clone()
        name_to_node_map = {op.get_friendly_name(): op for op in model.get_ops()}
        nodes_to_track_batch = [
            TrackedNodeInfo(name_to_node_map[node_name])
            for node_name in nodes_to_track_names[i : i + batch_size]
        ]

        # Add outputs for non-constant inputs of tracked nodes
        insert_outputs_for_tracked_ops(model, nodes_to_track_batch)
        # Infer model to collect tracked operation results and results of their inputs in full precision
        infer_full_net(nodes_to_track_batch, model, example_input)
        # Infer nodes in half precision one by one using full precision inputs, collect half precision results
        infer_nodes(nodes_to_track_batch, device, half_type)

        # Compute operation SNR based on full precision and half precision results
        for node_info in nodes_to_track_batch:
            try:
                snr = compute_snr(
                    node_info.node_value_full_precision,
                    node_info.node_value_half_precision,
                )
            except RuntimeError as e:
                # TODO: find the reason behind this
                if node_info.node.get_type_name() in [
                    "Add",
                    "Concat",
                ] and "Shape mismatch" in str(e):
                    print(
                        "Warning.",
                        str(e),
                        node_info.node.get_friendly_name(),
                        node_info.node.get_type_name(),
                        [
                            (inp_node.get_friendly_name(), inp_node.get_type_name())
                            for inp_node in node_info.input_nodes
                        ],
                    )
                    snr = np.finfo(np.float32).max
                else:
                    raise e
            node_names_and_snrs.append((node_info.node.get_friendly_name(), snr))

    if upcast_ratio != 0.0 and upcast_ratio != 1.0:
        node_names_and_snrs = sorted(node_names_and_snrs, key=lambda it: it[1])
        node_names, node_snrs = tuple(zip(*node_names_and_snrs))

        n_nodes = len(node_names)
        nodes_to_upcast_cnt = int(np.ceil(n_nodes * upcast_ratio))
        node_to_upcast_names = node_names[:nodes_to_upcast_cnt]

        if verbose:
            snr_quantile = node_snrs[nodes_to_upcast_cnt - 1]
            print(
                f"Upcasted {nodes_to_upcast_cnt}/{n_nodes} nodes with SNR less than {snr_quantile:.2f}."
            )
            for node_name, node_snr in node_names_and_snrs[:nodes_to_upcast_cnt]:
                print(node_name, node_snr)
    elif upcast_ratio == 0.0:
        if verbose:
            print(
                "Skipping algorithm because upcast ratio equals 0.0. Nothing to upcast."
            )
        node_to_upcast_names = []
    else:
        if verbose:
            print(
                "Skipping algorithm because upcast ratio equals 1.0. Upcasting all nodes of the given type(s)."
            )
        node_to_upcast_names = nodes_to_track_names

    new_model = orig_model.clone()
    mark_nodes_to_upcast_to_fp32(new_model, node_to_upcast_names)
    return new_model


def get_nodes_to_track(model: Model, operation_types: List[str]) -> List:
    nodes_to_track = []
    for i, op in enumerate(model.get_ordered_ops()):
        if op.get_type_name() in operation_types and all(
            map(
                lambda input: input.get_node().get_type_name() != "Result",
                op.output(0).get_target_inputs(),
            )
        ):
            nodes_to_track.append(op.get_friendly_name())
    return nodes_to_track


def insert_outputs_for_tracked_ops(
    model: Model, nodes_to_track: List[TrackedNodeInfo]
) -> None:
    node_to_output_map = OrderedDict()
    node_to_node_info_map = defaultdict(list)
    for node_info in nodes_to_track:
        node = node_info.node
        node_to_node_info_map[node].append(
            (node_info, "parent")
        )  # add as a parent node
        if node not in node_to_output_map:
            node_to_output_map[node] = node.output(0)
        node_info.input_nodes = []
        for inp_value in node.input_values():
            child_node = inp_value.get_node()
            node_info.input_nodes.append(child_node)
            # Do not add outputs for constant nodes
            if child_node.get_type_name() != "Constant" and not is_constant_path(
                child_node
            ):
                node_to_node_info_map[child_node].append(
                    (node_info, "child")
                )  # add as a child node
                if child_node not in node_to_output_map:
                    node_to_output_map[child_node] = child_node.output(0)

    outputs = model.add_outputs(list(node_to_output_map.values()))
    for output, node in zip(outputs, node_to_output_map.keys()):
        # Value matching will be done later based on result node friendly names
        result_node = output.node
        for node_info, parent_label in node_to_node_info_map[node]:
            is_parent = parent_label == "parent"
            if is_parent:
                node_info.result_node = result_node
            else:
                if node_info.input_result_nodes is None:
                    node_info.input_result_nodes = {}
                node_info.input_result_nodes[node] = result_node


def get_const_value_from_ovmodel(node: Union[Constant, Node]) -> np.ndarray:
    if node.get_type_name() == "Constant":
        assert node.get_element_type() not in [
            ov.Type.f16,
            ov.Type.bf16,
        ], f"{node.get_friendly_name()}, {node.get_element_type()}"
        return node.get_data()
    elif is_constant_path(node):
        # If model is compressed and constant values flow through decompression convert
        const_node = node.input_value(0).get_node()
        assert const_node.get_type_name() == "Constant"
        assert const_node.get_element_type().is_real(), const_node.get_element_type()
        return node.input_value(0).get_node().get_data()  # return f16 weight
    else:
        raise Exception(
            f"Cannot get const values from ov.Model for {node.get_friendly_name()} with type {node.get_type_name()}"
        )


def is_constant_path(node: Node) -> bool:
    if node.get_type_name() != "Convert":
        return False
    if len(node.get_rt_info()["is_decompression_0"].aslist()) > 0:
        return True
    if node.input_value(0).get_node().get_type_name() == "Constant":
        return True
    return False


def infer_full_net(
    nodes_to_track: List[TrackedNodeInfo], orig_model: Model, example_inputs: List
) -> None:
    core = ov.Core()
    exec_net = core.compile_model(
        orig_model, "CPU", config={"INFERENCE_PRECISION_HINT": "f32"}
    )
    request = exec_net.create_infer_request()
    results = request.infer(example_inputs, share_inputs=True, share_outputs=True)

    friendly_name_to_result_map = {}
    for i, (key, val) in enumerate(results.items()):
        result_node = key.node
        friendly_name_to_result_map[result_node.get_friendly_name()] = val

    for node_info in nodes_to_track:
        node_info.node_value_full_precision = friendly_name_to_result_map[
            node_info.result_node.get_friendly_name()
        ]
        node_info.input_values_full_precision = []
        for input_node in node_info.input_nodes:
            if input_node.get_type_name() == "Constant" or is_constant_path(input_node):
                # If input is constant, retrieve its value from model
                input_value = get_const_value_from_ovmodel(input_node)
            else:
                # If input is not constant, retrieve its input from inference results
                input_value = friendly_name_to_result_map[
                    node_info.input_result_nodes[input_node].get_friendly_name()
                ]
            node_info.input_values_full_precision.append(input_value)


def infer_nodes(
    nodes_to_track: List[TrackedNodeInfo], device: str, precision: str
) -> None:
    for node_info in nodes_to_track:
        infer_tracked_op(node_info, device, precision)


def infer_tracked_op(node_info: TrackedNodeInfo, device: str, precision: str) -> None:
    parameters = []
    inputs = []
    input_values = node_info.input_values_full_precision
    for input_value in input_values:
        parameter = Parameter(
            get_element_type(input_value.dtype), ov.PartialShape(input_value.shape)
        )
        if input_value.dtype == np.float16:
            # Convert f16 weight to f32
            convert_node = opset.convert(parameter, "f32")
            inputs.append(convert_node)
        else:
            inputs.append(parameter)
        parameters.append(parameter)

    node = node_info.node
    try:
        call_attributes = node.get_attributes()
        # Below are some op workarounds
        if node.get_type_name() == "Divide" and "m_pythondiv" in call_attributes:
            del call_attributes["m_pythondiv"]
        if node.get_type_name() == "Broadcast" and "mode" in call_attributes:
            call_attributes["broadcast_spec"] = call_attributes["mode"]
            del call_attributes["mode"]
        if node.get_type_name() == "Concat":
            new_op = OPERATION_TYPE_MAP[node.get_type_name()](inputs, **call_attributes)
        else:
            new_op = OPERATION_TYPE_MAP[node.get_type_name()](
                *inputs, **call_attributes
            )

        ov_model = ov.Model([new_op], parameters=parameters)
        exec_net = ov.Core().compile_model(
            ov_model, device, config={"INFERENCE_PRECISION_HINT": precision}
        )
        request = exec_net.create_infer_request()
        result = request.infer(input_values, share_inputs=True, share_outputs=True)
    except Exception as e:
        print(
            "Operation inference error",
            node.get_type_name(),
            node.get_friendly_name(),
            inputs,
            node.get_attributes(),
        )
        raise e

    node_info.node_value_half_precision = result[0]
    assert len(result) == 1


def is_model_partially_upcasted(model) -> bool:
    for node in model.get_ordered_ops():
        if node.get_type_name() not in OPERATION_TYPE_MAP.keys():
            continue
        if ORIGINAL_PRECISION_RT_INFO_NAME in node.get_rt_info().keys():
            return True
    return False


def mark_nodes_to_upcast_to_fp32(model: ov.Model, nodes_with_errors: List[str]) -> None:
    nodes_to_mark = set(nodes_with_errors)
    for node in model.get_ordered_ops():
        if node.get_friendly_name() in nodes_to_mark:
            node.get_rt_info()[ORIGINAL_PRECISION_RT_INFO_NAME] = ""
            nodes_to_mark.remove(node.get_friendly_name())
    assert len(nodes_to_mark) == 0, nodes_to_mark


def compute_snr(x, y):
    # x -- original value (full precision), y -- value with noise (half precision)

    x, y = x.astype(np.float32), y.astype(np.float32)
    max_value = np.finfo(np.float32).max

    if np.prod(x.shape) != np.prod(y.shape):
        raise RuntimeError(f"Shape mismatch: {x.shape}, {y.shape}.")

    x = np.nan_to_num(x, posinf=max_value)
    y = np.nan_to_num(y, posinf=max_value)

    Ps = np.linalg.norm(x)
    Pn = np.nan_to_num(np.linalg.norm(x - y), posinf=max_value)

    if Ps == Pn == 0.0:
        return max_value

    snr = np.nan_to_num(20 * np.log10(Ps / Pn), posinf=max_value)

    return snr
