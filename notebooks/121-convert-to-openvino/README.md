# OpenVINO™ IR conversion API tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F121-convert-to-openvino%2F121-convert-to-openvino.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/121-convert-to-openvino/121-convert-to-openvino.ipynb)

This notebook shows how to convert model from original framework format to the OpenVINO IR. It describes Python conversion API and Model Optimizer command-line tool. It provides examples of Hugging Face and Pytorch models conversion to OpenVINO IR.

## Notebook Contents

The OpenVINO IR conversion tutorial consists of the following content:

* OpenVINO IR format
* IR preparation with Python conversion API and Model Optimizer command-line tool
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