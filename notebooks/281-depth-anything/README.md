 Monodepth Demo

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F281-depth-anythingh%2F281-depth-anything.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/281-depth-anything/281-depth-anything.ipynb)

![depth_map.png](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/a9a16658-512f-470c-a33c-0e1f9d0ae72c)

[Depth Anything](https://depth-anything.github.io/) is a highly practical solution for robust monocular depth estimation. Without pursuing novel technical modules, this project aims to build a simple yet powerful foundation model dealing with any images under any circumstances.
The framework of Depth Anything is shown below. it adopts a standard pipeline to unleashing the power of large-scale unlabeled images. 
![image.png](attachment:df8f001d-8132-4ea5-bec2-b6b9f55089a1.png)

More details about model can be found in [project webpage](https://depth-anything.github.io/), [paper](https://arxiv.org/abs/2401.10891), and official [repository](https://github.com/LiheYoung/Depth-Anything)

In this tutorial we will explore how to convert and run DepthAnything using OpenVINO.

## Notebook Contents

This notebook demonstrates Monocular Depth Estimation with the [DepthAnything](https://github.com/LiheYoung/Depth-Anything) in OpenVINO.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
