#include <torch/extension.h>
#include "voxelization.h"

namespace voxelization {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hard_voxelize", &hard_voxelize, "hard voxelize");
}

} // namespace voxelization
