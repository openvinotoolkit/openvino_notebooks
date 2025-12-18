// OpenVINO Extension library entry point
// Registers custom operations for PointPillars

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include "voxelization_op.hpp"
#include "postprocessing_op.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<ov::custom_ops::VoxelizationOp>>(),
        std::make_shared<ov::OpExtension<ov::custom_ops::PostProcessingOp>>()
    }));
