// OpenVINO custom operation for PointPillars voxelization

#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace custom_ops {

class VoxelizationOp : public ov::op::Op {
public:
    OPENVINO_OP("VoxelizationOp");

    VoxelizationOp() = default;

    VoxelizationOp(const ov::Output<ov::Node>& points,
                   const std::vector<float>& voxel_size,
                   const std::vector<float>& point_cloud_range,
                   int max_points_per_voxel,
                   int max_voxels);

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override;

private:
    std::vector<float> m_voxel_size;           // [vx, vy, vz]
    std::vector<float> m_point_cloud_range;    // [x_min, y_min, z_min, x_max, y_max, z_max]
    int m_max_points_per_voxel;
    int m_max_voxels;
};

}  // namespace custom_ops
}  // namespace ov
