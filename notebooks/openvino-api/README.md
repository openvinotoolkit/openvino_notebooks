# OpenVINO™ API tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fopenvino-api%2Fopenvino-api.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/openvino-api/openvino-api.ipynb)


This notebook explains the basics of the OpenVINO Runtime API.
It provides a segmentation and classification IR model and a segmentation ONNX model. The model files can be replaced with your own models.

Despite the exact output being different, the process remains the same.

## Notebook Contents

The OpenVINO API tutorial consists of the following steps:

* Loading OpenVINO Runtime and Showing Info
* Loading a Model
  * OpenVINO IR Model
  * ONNX Model
  * PaddlePaddle Model
  * TensorFlow Model
  * TensorFlow Lite Model
  * PyTorch Model
* Getting Information about a Model
  * Model Inputs
  * Model Outputs
* Doing Inference on a Model
* Reshaping and Resizing
  * Change Image Size
  * Change Batch Size
  
## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
