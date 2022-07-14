# Super Resolution with PaddleGAN and OpenVINO

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F207-vision-paddlegan-superresolution%2F207-vision-paddlegan-superresolution.ipynb)

This notebook demonstrates converting the RealSR (real-world super-resolution) model from [PaddlePaddle/PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN) to OpenVINO Intermediate Representation (OpenVINO IR) format, and shows inference results on both the PaddleGAN and OpenVINO IR models. 

For more information about the various PaddleGAN superresolution models, refer to the [PaddleGAN documentation](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/tutorials/single_image_super_resolution.md). For more information about RealSR, see the [research paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf) from CVPR 2020.

This notebook works best with small images (up to 800x600 resolution).

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
