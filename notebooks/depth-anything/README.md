# Depth estimation with DepthAnything and OpenVINO

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fdepth-anythingh%2Fdepth-anything.ipynb)

![depth_map.gif](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/a9a16658-512f-470c-a33c-0e1f9d0ae72c)


DepthAnything is a highly practical solution for robust monocular depth estimation. Without pursuing novel technical modules, this project aims to build a simple yet powerful foundation model dealing with any images under any circumstances.
The framework of Depth Anything is shown below. it adopts a standard pipeline to unleashing the power of large-scale unlabeled images. 
![image.png](https://depth-anything.github.io/static/images/pipeline.png)

There are two version of DepthAnything models. The notebooks series include the demonstration of work [Depth Anything V1](https://depth-anything.github.io/) and [Depth Anything V2](https://depth-anything.github.io/). Depth Anything V2 significantly outperforms V1 in fine-grained details and robustness.

* [Depth Anything V1 with OpenVINO](./depth-anything.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/depth-anything/depth-anything.ipynb)

* [Depth Anything V2 with OpenVINO](./depth-anything-v2.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/depth-anything/depth-anything-v2.ipynb)


In these tutorials we will explore how to convert and run DepthAnything using OpenVINO. An additional part demonstrates how to run quantization with [NNCF](https://github.com/openvinotoolkit/nncf/) to speed up the model.

## Notebook Contents

Both tutorials consists of following steps:
- Install prerequisites
- Load and run PyTorch model inference
- Convert Model to Openvino Intermediate Representation format
- Run OpenVINO model inference on single image
- Run OpenVINO model inference on video
- Optimize Model
- Compare results of original and optimized models
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/depth-anything/README.md" />
