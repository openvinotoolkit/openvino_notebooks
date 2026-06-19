# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import Iterable

import numpy as np
import openvino as ov
from openvino import PartialShape, Type
from openvino import opset15 as opset
from openvino.passes import Manager, ModelPass
from openvino.utils.types import make_constant_node

from . import _get_factory
from .constants import PROCESSED_POST_PROCESSOR_NAME


logger = logging.getLogger(__name__)


class ModifyCombineSegmentsForPairInput(ModelPass):
    """
    Changes the ov.Model so that it accepts paired input.

    Concatenate both input, process, tokenize and then split them near the end before CombineSegments node.
    If any constant SpecialToken depends on sequence inputs, thay are zeroed.
    Truncation operation is also modified to support max_length.
    """

    def parse_inputs(self, combine_seg: ov.Node) -> bool:
        """
        Get combine segment inputs, and input_signature.
        The main sequence inputs should go through node, eg. Truncate -> CombineSegments
        """
        num_segments = int(combine_seg.get_input_size() - 1) // 3

        # We go through the inputs of the CombineSegments node and check if they are
        # either Constant or Sequence. If begin is Constant, we check if the corresponding
        # end input is also Constant.
        # If begin is Sequence, we check if input has truncation operation.
        inputs = []
        input_signature = [[]] * num_segments
        for i in range(num_segments):
            if isinstance(combine_seg.input_value(3 * i).node, ov.op.Constant):
                # Constant input
                if not isinstance(combine_seg.input_value(3 * i + 2).node, ov.op.Constant):
                    return False
                input_signature[i] = combine_seg.input_value(3 * i + 2).node.get_data().item(0)
                inputs.extend(
                    [
                        combine_seg.input_value(3 * i),
                        combine_seg.input_value(3 * i + 1),
                        combine_seg.input_value(3 * i + 2),
                    ]
                )
            else:
                # Sequence input
                input_signature[i] = -1

                # Input to CombineSegments node should contain Truncate operation
                trunc_node = combine_seg.input_value(3 * i).node
                if trunc_node.get_type_name() != "Truncate":
                    return False

                inputs.extend(trunc_node.input_values()[:3])
                self.trunc_values = trunc_node.input_values()[3:]
        self.inputs = inputs
        self.input_signature = input_signature
        return True

    def insert_splits(self):
        """
        Insert Splits for begins, ends before the CombineSegments node and returns new inputs.
        Also adds a modified Truncate
        """
        inputs = self.inputs
        input_signature = self.input_signature

        first_input_idx = input_signature.index(-1)
        begin, end, data = inputs[3 * first_input_idx : 3 * first_input_idx + 3]

        # Single parameter will be replaced with a concatenated pair of Parameters.
        new_parameters = [ov.op.Parameter(Type.string, PartialShape([-1])) for i in range(2)]
        for i in range(2):
            new_parameters[i].set_friendly_name(f"string_input_{i + 1}")

        param_1_shape = opset.shape_of(new_parameters[0], output_type="i32")
        param_2_shape = opset.shape_of(new_parameters[1], output_type="i32")
        total_size = opset.shape_of(begin, output_type="i32")
        self.new_parameters = new_parameters

        # For the first input begins_1/ends_1, it's a slice till the Parameter_1 shape.
        begins_1 = opset.slice(begin, start=[0], stop=param_1_shape, step=[1], name="begins_1")
        ends_1 = opset.slice(end, start=[0], stop=param_1_shape, step=[1], name="ends_1")

        # If the second input is empty we need to slice at least one element.
        # If we don't do that input with shape [0] could not be specified together with input with shape [1]
        # in Select and broadcasted. This garbage values will be zeroed in Select.
        second_start = opset.minimum(
            total_size - param_2_shape, total_size - make_constant_node([1], Type.i32), name="start_for_second"
        )

        # For the second input begins_2/ends_2, slice till the end.
        begins_2 = opset.slice(begin, start=second_start, stop=total_size, step=[1], name="begins_2")
        ends_2 = opset.slice(end, start=second_start, stop=total_size, step=[1], name="ends_2")

        # data is left unchanged for both inputs

        # If inputs_2 is empty, we need to zero the second dimension of the broadcasted begins and ends tensors.
        # TODO: indeed for ends it should've been 1 but for thix bug CSV-160624 we set to zero for the moment.
        self.equal_node = opset.equal(param_2_shape, make_constant_node([0], Type.i32), name="is_paired_input")
        begins_2 = opset.select(self.equal_node, make_constant_node([0], Type.i32), begins_2)
        ends_2 = opset.select(self.equal_node, make_constant_node([0], Type.i32), ends_2)

        # broadcast the begins and ends tensors so that if there is one candidate and
        # a batch of query inputs (or vice versa) they are broadcasted to the same shape.
        broadcasted_shape = opset.maximum(param_1_shape, param_2_shape, name="broadcasted_shape")
        first_input = [
            opset.broadcast(begins_1, broadcasted_shape).output(0),
            opset.broadcast(ends_1, broadcasted_shape).output(0),
            data,
        ]
        second_input = [
            opset.broadcast(begins_2, broadcasted_shape).output(0),
            opset.broadcast(ends_2, broadcasted_shape).output(0),
            data,
        ]

        signature_to_extend = self.post_processor["pair"]["ids"][len(input_signature) :]

        # Since we add additional special tokens the max_length for truncation should be reduced.
        self.trunc_values[0] = make_constant_node(
            self.trunc_values[0].node.data - (len(signature_to_extend) - 1), Type.i32
        ).output(0)

        trunc = _get_factory().create("Truncate", [*first_input, *second_input, *self.trunc_values])
        first_input, second_input = trunc.outputs()[:3], trunc.outputs()[3:6]
        self.first_input = first_input
        self.second_input = second_input

    def get_new_inputs(self) -> list:
        """
        Creates new inputs for the CombineSegments node.
        It combines inputs for the first and second input, and adds special tokens for the second input.
        The new inputs are then returned as a list.
        """
        inputs = self.inputs
        input_signature = self.input_signature
        first_input = self.first_input
        second_input = self.second_input

        new_inputs = inputs.copy()
        # Replace original input with concatenated Parameter_1, Parameter_2 with only Parameter_1 input.
        # [bos, concat(sequence_1, sequnce_2), eos] -> [bos, sequence_1, eos]
        first_input_idx = input_signature.index(-1)
        new_inputs[3 * first_input_idx : 3 * first_input_idx + 3] = first_input

        #  if original input_signature [bos, sequence_1, eos]
        #  if pair_signature [bos, sequence_1, eos, sequence_2, eos]
        # then singature_to_extend = [sequence_2, eos]
        signature_to_extend = self.post_processor["pair"]["ids"][len(input_signature) :]
        for value in signature_to_extend:
            if value <= -1:
                # we ensured previous that only one additional input is possible
                new_inputs.extend(second_input)
                continue

            if not isinstance(value, Iterable):
                value = [value]
            if not all(isinstance(item, int) for item in value):
                # input signature ids should be value or a list of values
                return False

            added_spec_begins = make_constant_node(0, Type.i32).output(0)
            added_spec_ends = make_constant_node(len(value), Type.i32).output(0)
            added_spec_data = make_constant_node(value, Type.i32).output(0)

            # If ends for the sequence_2 is nullified, we should nullify special_tokens constant as well
            added_spec_ends = opset.multiply(
                added_spec_ends,
                opset.select(self.equal_node, make_constant_node(0, Type.i32), make_constant_node(1, Type.i32)),
            ).output(0)
            new_spec_tokens = [added_spec_begins, added_spec_ends, added_spec_data]

            new_inputs.extend(new_spec_tokens)

        # For the added inputs segment ids should be 1.
        new_segment_ids = make_constant_node(self.post_processor["pair"]["type_ids"], Type.i32).output(0)
        new_inputs.append(new_segment_ids)
        return new_inputs

    def assert_and_get_postprocessor(
        self,
        model: ov.Model,
    ):
        if PROCESSED_POST_PROCESSOR_NAME not in model.rt_info:
            logger.info("Could not add second input. Post processor is not present in the model.")
            return False
        post_processor = json.loads(model.get_rt_info(PROCESSED_POST_PROCESSOR_NAME).value)

        if "pair" not in post_processor:
            logger.info("Could not add second input. post_processor does not contain input signature for paired input")
            return False

        if post_processor["single"]["ids"] != self.input_signature:
            logger.info(
                "Could not add second input. Input signature from rt_info does not match to the CombineSegments node inputs."
            )

        if post_processor["pair"]["ids"][: len(self.input_signature)] != self.input_signature:
            logger.info(
                "Could not add second input. Paried inputs are allowed only when it's widening the single input."
            )
            return False

        if len(np.nonzero(np.array(post_processor["pair"]["ids"]) <= -1)[0]) != 2:
            logger.info("Could not add second input. Only 2 inputs are allowed for the paired input")
            return False

        if len(np.nonzero(np.array(post_processor["single"]["ids"]) <= -1)[0]) != 1:
            logger.info(
                "Could not add second input. There should be exactly one sequence input in the single signature."
            )
            return False
        self.post_processor = post_processor
        return True

    def run_on_model(self, model: ov.Model):
        parameters = model.get_parameters()
        if len(parameters) != 1:
            logger.info(
                f"Could not add second input. Original model should have only one input, while it has {len(parameters)}"
            )
            return False

        # Find the CombineSegments node in the model
        combine_seg = next((op for op in model.get_ops() if op.get_type_name() == "CombineSegments"), None)
        if not combine_seg:
            logger.info("Could not add second input. Original model does not contain CombineSegments node.")
            return False

        # Check if the CombineSegments node has the expected input signature
        # and save the input signature in self.input_signature.
        # If signature is incorrect we exit the transformation without modifying ov.Model.
        if not self.parse_inputs(combine_seg):
            return False

        # Check if the model has a post-processor and if it contains the expected input signature
        if not self.assert_and_get_postprocessor(model):
            return False

        # Insert Splits for begins, ends before the CombineSegments node and returns pair of new inputs.
        # Also adds a modified Truncate
        self.insert_splits()

        # Get new inputs for the CombineSegments node, which combine
        # the first and second input, and add special tokens for the second input.
        new_inputs = self.get_new_inputs()
        if not new_inputs:
            return False

        # Replace the CombineSegments node with a new one that takes the pair of inputs
        new_combine_segments = _get_factory().create("CombineSegments", new_inputs)
        ov.utils.replace_node(combine_seg, new_combine_segments)

        target_inputs = list(parameters[0].output(0).get_target_inputs())
        str_unpack: ov.Node = target_inputs[0].get_node()
        if str_unpack.get_type_name() != "StringTensorUnpack":
            return False

        new_input = opset.concat(self.new_parameters, axis=0)
        str_unpack.input(0).replace_source_output(new_input.output(0))

        model.replace_parameter(0, self.new_parameters[0])
        model.add_parameters([self.new_parameters[1]])

        return True


def add_second_input(model: ov.Model):
    """
    Extends inplace the input of the model to a pair of inputs.
    """
    manager = Manager()
    manager.register_pass(ModifyCombineSegmentsForPairInput())
    manager.run_passes(model)
