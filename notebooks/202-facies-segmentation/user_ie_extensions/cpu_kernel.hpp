// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include <ngraph/ngraph.hpp>

namespace TemplateExtension {

//! [cpu_implementation:header]
class OpImplementation : public InferenceEngine::ILayerExecImpl {
public:
    explicit OpImplementation(const std::shared_ptr<ngraph::Node>& node);
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
                                                           InferenceEngine::ResponseDesc *resp) noexcept override;
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig &config,
                                     InferenceEngine::ResponseDesc *resp) noexcept override;
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                        std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                        InferenceEngine::ResponseDesc *resp) noexcept override;
private:
    std::vector<ngraph::Shape> inShapes;
    ngraph::Shape outShape;
    std::string error;
    std::vector<bool> mask;
};
//! [cpu_implementation:header]

}  // namespace TemplateExtension
