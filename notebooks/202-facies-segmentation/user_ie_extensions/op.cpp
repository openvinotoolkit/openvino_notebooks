// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op.hpp"

using namespace TemplateExtension;

constexpr ngraph::NodeTypeInfo Operation::type_info;

//! [op:ctor]
Operation::Operation(const ngraph::Output<ngraph::Node>& poolInp,
                     const ngraph::Output<ngraph::Node>& poolOut,
                     const ngraph::Output<ngraph::Node>& inp,
                     const ngraph::Output<ngraph::Node>& shape) : Op({poolInp, poolOut, inp, shape}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void Operation::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(3);
    auto poolInpShape = get_input_partial_shape(0).to_shape();
    outShape[0] = poolInpShape[0];  // Use only spatial dimensions from shape
    outShape[1] = poolInpShape[1];  // and restore batch and channels
    set_output_type(0, get_input_element_type(0), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ngraph::Node> Operation::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    if (new_args.size() != 4) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<Operation>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}
//! [op:copy]

//! [op:visit_attributes]
bool Operation::visit_attributes(ngraph::AttributeVisitor &visitor) {
    return true;
}
//! [op:visit_attributes]
