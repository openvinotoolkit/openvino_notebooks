// OpenVINO custom operation for PointPillars voxelization

#include "voxelization_op.hpp"
#include <openvino/core/except.hpp>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace ov {
namespace custom_ops {

VoxelizationOp::VoxelizationOp(const ov::Output<ov::Node>& points,
                               const std::vector<float>& voxel_size,
                               const std::vector<float>& point_cloud_range,
                               int max_points_per_voxel,
                               int max_voxels)
    : ov::op::Op({points}),
      m_voxel_size(voxel_size),
      m_point_cloud_range(point_cloud_range),
      m_max_points_per_voxel(max_points_per_voxel),
      m_max_voxels(max_voxels) {
    constructor_validate_and_infer_types();
}

void VoxelizationOp::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 1, "VoxelizationOp expects 1 input");

    const auto& points_shape = get_input_partial_shape(0);
    OPENVINO_ASSERT(points_shape.rank().is_static() && points_shape.rank().get_length() == 2,
                    "Points input must be 2D [num_points, num_features]");

    // Output shapes:
    // voxels: [-1, max_points_per_voxel, num_features]  (dynamic voxel count)
    // coors: [-1, 4]  (batch_idx, x, y, z indices)
    // num_points_per_voxel: [-1]

    const auto num_features = points_shape[1];

    set_output_type(0, get_input_element_type(0),
                    ov::PartialShape{ov::Dimension::dynamic(),
                                     m_max_points_per_voxel, num_features});
    set_output_type(1, ov::element::i64,
                    ov::PartialShape{ov::Dimension::dynamic(), 4});
    set_output_type(2, ov::element::i64,
                    ov::PartialShape{ov::Dimension::dynamic()});
}

std::shared_ptr<ov::Node> VoxelizationOp::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

    return std::make_shared<VoxelizationOp>(new_args[0],
                                             m_voxel_size,
                                             m_point_cloud_range,
                                             m_max_points_per_voxel,
                                             m_max_voxels);
}

bool VoxelizationOp::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("voxel_size", m_voxel_size);
    visitor.on_attribute("point_cloud_range", m_point_cloud_range);
    visitor.on_attribute("max_points_per_voxel", m_max_points_per_voxel);
    visitor.on_attribute("max_voxels", m_max_voxels);
    return true;
}

bool VoxelizationOp::has_evaluate() const {
    return true;
}

bool VoxelizationOp::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 1, "VoxelizationOp expects 1 input");
    OPENVINO_ASSERT(outputs.size() == 3, "VoxelizationOp expects 3 outputs");

    const auto& points_tensor = inputs[0];
    const auto points_shape = points_tensor.get_shape();
    const int num_points = points_shape[0];
    const int num_features = points_shape[1];

    const float* points_data = points_tensor.data<float>();

    // Prepare outputs - use temporary buffers with max size
    auto& voxels_tensor = outputs[0];
    auto& coors_tensor = outputs[1];
    auto& num_points_tensor = outputs[2];

    // Allocate temporary buffers with max_voxels size
    std::vector<float> voxels_buf(
        m_max_voxels * m_max_points_per_voxel * num_features, 0.0f);
    std::vector<int64_t> coors_buf(m_max_voxels * 4,
                               0); // 4 columns: batch_idx, x, y, z
    std::vector<int64_t> num_points_buf(m_max_voxels, 0);

    float *voxels_data = voxels_buf.data();
    int64_t *coors_data = coors_buf.data();
    int64_t *num_points_data = num_points_buf.data();

    // Grid dimensions
    const float vx = m_voxel_size[0];
    const float vy = m_voxel_size[1];
    const float vz = m_voxel_size[2];

    const float x_min = m_point_cloud_range[0];
    const float y_min = m_point_cloud_range[1];
    const float z_min = m_point_cloud_range[2];
    const float x_max = m_point_cloud_range[3];
    const float y_max = m_point_cloud_range[4];
    const float z_max = m_point_cloud_range[5];

    const int grid_x = std::round((x_max - x_min) / vx);
    const int grid_y = std::round((y_max - y_min) / vy);
    const int grid_z = std::round((z_max - z_min) / vz);

    // Hash map to track voxel indices
    std::unordered_map<int, int> coor_to_voxelidx;
    int voxel_num = 0;

    for (int i = 0; i < num_points; ++i) {
        const float* point = points_data + i * num_features;
        const float px = point[0];
        const float py = point[1];
        const float pz = point[2];

        // Check if point is in range
        if (px < x_min || px >= x_max || py < y_min || py >= y_max ||
            pz < z_min || pz >= z_max) {
            continue;
        }

        // Compute voxel coordinates
        int c_x = std::floor((px - x_min) / vx);
        int c_y = std::floor((py - y_min) / vy);
        int c_z = std::floor((pz - z_min) / vz);

        // Clamp to grid
        c_x = std::max(0, std::min(c_x, grid_x - 1));
        c_y = std::max(0, std::min(c_y, grid_y - 1));
        c_z = std::max(0, std::min(c_z, grid_z - 1));

        // Compute hash (linearize coordinates)
        int coor_hash = c_z * grid_y * grid_x + c_y * grid_x + c_x;

        int voxel_idx;
        auto it = coor_to_voxelidx.find(coor_hash);
        if (it == coor_to_voxelidx.end()) {
            // New voxel
            if (voxel_num >= m_max_voxels) {
                continue; // Max voxels reached
            }
            voxel_idx = voxel_num;
            coor_to_voxelidx[coor_hash] = voxel_idx;

            coors_data[voxel_idx * 4 + 0] =
                0; // batch index (always 0 for single sample)
            coors_data[voxel_idx * 4 + 1] = c_x;
            coors_data[voxel_idx * 4 + 2] = c_y;
            coors_data[voxel_idx * 4 + 3] = c_z;

            voxel_num++;
        } else {
            voxel_idx = it->second;
        }

        // Add point to voxel
        int64_t& num_pts = num_points_data[voxel_idx];
        if (num_pts < m_max_points_per_voxel) {
            float* voxel_point = voxels_data +
                                 (voxel_idx * m_max_points_per_voxel + num_pts) * num_features;
            std::memcpy(voxel_point, point, num_features * sizeof(float));
            num_pts++;
        }
    }

    // std::cerr << "[VoxelizationOp] voxel_num=" << voxel_num
    //           << ", m_max_voxels=" << m_max_voxels << std::endl;

    voxels_tensor.set_shape({static_cast<size_t>(voxel_num),
                             static_cast<size_t>(m_max_points_per_voxel),
                             static_cast<size_t>(num_features)});
    coors_tensor.set_shape({static_cast<size_t>(voxel_num), 4});
    num_points_tensor.set_shape({static_cast<size_t>(voxel_num)});

    // Copy only the valid voxel data to output tensors
    std::memcpy(voxels_tensor.data<float>(), voxels_buf.data(),
                voxel_num * m_max_points_per_voxel * num_features *
                    sizeof(float));
    std::memcpy(coors_tensor.data<int64_t>(), coors_buf.data(),
                voxel_num * 4 * sizeof(int64_t));
    std::memcpy(num_points_tensor.data<int64_t>(), num_points_buf.data(),
                voxel_num * sizeof(int64_t));

    return true;
}

}  // namespace custom_ops
}  // namespace ov
