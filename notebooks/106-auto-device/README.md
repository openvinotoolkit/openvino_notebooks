# Introduction to Auto Device Selection in OpenVINO

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F106-auto-device%2F106-auto-device.ipynb)

This notebook demonstrates how to do inference with Automatic Device Selection. More information about Automatic Device Selection: [click >>>](https://docs.openvino.ai/latest/openvino_docs_IE_DG_supported_plugins_AUTO.html)

## Notebook Contents

A basic introduction to do Auto Device Selection with OpenVINO. 

This notebook demostrate how to compile_model with AUTO device, compare the first inference latency (model compilation time + 1st inference time) between GPU device and AUTO device, show the difference performance hints (THROUGHPUT and LATENCY) with significant performance results towards the desired metric.

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
