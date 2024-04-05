# OpenVINO™ model conversion API tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fconvert-to-openvino%2Fconvert-to-openvino.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/convert-to-openvino/convert-to-openvino.ipynb)

This notebook shows how to convert a model from its original framework format to the OpenVINO IR. It describes the Python conversion API and the OpenVINO Model Converter command-line tool. It provides examples of converting Hugging Face and PyTorch models to OpenVINO IR.

## Notebook Contents

The OpenVINO IR conversion tutorial consists of the following content:

* OpenVINO IR format
* Fetching example models
* Conversion
    * Setting Input Shapes
    * Compressing a Model to FP16
    * Converting Models from Memory
* Migration from Legacy Conversion API
    * Specifying Layout
    * Changing Model Layout
    * Specifying Mean and Scale Values
    * Reversing Input Channels
    * Cutting Off Parts of a Model

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).