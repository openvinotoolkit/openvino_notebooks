# OpenVINO™ IR preparation tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F121-convert-to-openvino-ir%2F121-convert-to-openvino-ir.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/121-convert-to-openvino-ir/121-convert-to-openvino-ir.ipynb)

This notebook explains the basics of the OpenVINO model preparation API. It provides examples of Hugging Face and Pytorch models conversion to OpenVINO IR with various parameters.

## Notebook Contents

The OpenVINO IR preparation tutorial consists of the following content:

* OpenVINO IR format
* IR preparation with Python model conversion API and Model Optimizer command-line tool
* Fetching example models
* Basic conversion
* Model conversion parameters
  * Setting Input Shapes
  * Cutting Off Parts of a Model
  * Embedding Preprocessing Computation
    * Specifying Layout
    * Changing Model Layout
    * Specifying Mean and Scale Values
    * Reversing Input Channels
  * Compressing a Model to FP16
* Convert Models Represented as Python Objects

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).