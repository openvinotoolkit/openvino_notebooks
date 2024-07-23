# Super Resolution with PaddleGAN and OpenVINO

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/vision-paddlegan-superresolution/vision-paddlegan-superresolution.ipynb)

This notebook demonstrates converting the RealSR (real-world super-resolution) model from [PaddlePaddle/PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN) to OpenVINO Intermediate Representation (OpenVINO IR) format, and shows inference results on both the PaddleGAN and OpenVINO IR models. 

For more information about the various PaddleGAN superresolution models, refer to the [PaddleGAN documentation](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/tutorials/single_image_super_resolution.md). For more information about RealSR, see the [research paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf) from CVPR 2020.

This notebook works best with small images (up to 800x600 resolution).

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
