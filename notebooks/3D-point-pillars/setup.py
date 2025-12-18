import os

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

def make_extensions():
  cpu_build = os.environ.get("CPU_BUILD", "0") == "1"

  if cpu_build:
    print("CPU extension")
    from torch.utils.cpp_extension import CppExtension
    return [
      CppExtension(
        name='pointpillars.ops.voxel_op',
        sources=[
          'pointpillars/ops/voxelization/voxelization.cpp',
          'pointpillars/ops/voxelization/voxelization_cpu.cpp',
        ],
      ),
      CppExtension(
        name='pointpillars.ops.iou3d_op',
        sources=[
          'pointpillars/ops/iou3d/iou3d.cpp',
          'pointpillars/ops/iou3d/iou3d_kernel_cpu.cpp',
        ],
      ),
    ]
  else: # cuda build
    print("CUDA extension")
    from torch.utils.cpp_extension import CUDAExtension
    return [
      CUDAExtension(
        name='pointpillars.ops.voxel_op',
        sources=[
          'pointpillars/ops/voxelization/voxelization.cpp',
          'pointpillars/ops/voxelization/voxelization_cpu.cpp',
          'pointpillars/ops/voxelization/voxelization_cuda.cu',
        ],
        define_macros=[('WITH_CUDA', None)],
      ),
      CUDAExtension(
        name='pointpillars.ops.iou3d_op',
        sources=[
          'pointpillars/ops/iou3d/iou3d.cpp',
          'pointpillars/ops/iou3d/iou3d_kernel.cu',
        ],
        define_macros=[('WITH_CUDA', None)],
      ),
    ]

setup(
  name='pointpillars',
  version='0.1',
  packages=find_packages(),
  ext_modules=make_extensions(),
  cmdclass={'build_ext': BuildExtension},
  zip_safe=False
)
