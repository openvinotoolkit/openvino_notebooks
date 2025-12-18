#pragma once
#include <torch/extension.h>

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

namespace voxelization {

int hard_voxelize_cpu(const at::Tensor &points, at::Tensor &voxels,
                      at::Tensor &coors, at::Tensor &num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim = 3);

#ifdef WITH_CUDA
int hard_voxelize_gpu(const at::Tensor &points, at::Tensor &voxels,
                      at::Tensor &coors, at::Tensor &num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim = 3);

int nondisterministic_hard_voxelize_gpu(const at::Tensor &points, at::Tensor &voxels,
                                        at::Tensor &coors, at::Tensor &num_points_per_voxel,
                                        const std::vector<float> voxel_size,
                                        const std::vector<float> coors_range,
                                        const int max_points, const int max_voxels,
                                        const int NDim = 3);
#endif

// Interface for Python
inline int hard_voxelize(const at::Tensor &points, at::Tensor &voxels,
                         at::Tensor &coors, at::Tensor &num_points_per_voxel,
                         const std::vector<float> voxel_size,
                         const std::vector<float> coors_range,
                         const int max_points, const int max_voxels,
                         const int NDim = 3, const bool deterministic = true) {
  if (points.device().is_cuda()) {
#ifdef WITH_CUDA
    if (deterministic) {
      return hard_voxelize_gpu(points, voxels, coors, num_points_per_voxel,
                               voxel_size, coors_range, max_points, max_voxels,
                               NDim);
    }
    return nondisterministic_hard_voxelize_gpu(points, voxels, coors, num_points_per_voxel,
                                               voxel_size, coors_range, max_points, max_voxels,
                                               NDim);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return hard_voxelize_cpu(points, voxels, coors, num_points_per_voxel,
                           voxel_size, coors_range, max_points, max_voxels,
                           NDim);
}


inline reduce_t convert_reduce_type(const std::string &reduce_type) {
  if (reduce_type == "max")
    return reduce_t::MAX;
  else if (reduce_type == "sum")
    return reduce_t::SUM;
  else if (reduce_type == "mean")
    return reduce_t::MEAN;
  else TORCH_CHECK(false, "do not support reduce type " + reduce_type)
  return reduce_t::SUM;
}

}  // namespace voxelization
