# Introduction to Auto Device Selection in OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fauto-device%2Fauto-device.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/auto-device/auto-device.ipynb)

This notebook shows how to do inference with Automatic Device Selection (AUTO). To learn more about the logic of this mode, refer to the [Automatic device selection](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html) article.

## Notebook Contents

A basic introduction to use Auto Device Selection with OpenVINO.

This notebook demonstrates how to compile a model with AUTO device, compare the first inference latency (model compilation time + 1st inference time) between GPU device and AUTO device, show the difference in performance hints (THROUGHPUT and LATENCY) with significant performance results towards the desired metric.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
