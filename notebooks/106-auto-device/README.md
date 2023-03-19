# Introduction to Auto Device Selection in OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F106-auto-device%2F106-auto-device.ipynb)

This notebook shows how to do inference with Automatic Device Selection (AUTO). To learn more about the logic of this mode, refer to the [Automatic device selection](https://docs.openvino.ai/latest/openvino_docs_IE_DG_supported_plugins_AUTO.html) article.

## Notebook Contents

A basic introduction to use Auto Device Selection with OpenVINO. 

This notebook demonstrates how to compile a model with AUTO device, compare the first inference latency (model compilation time + 1st inference time) between GPU device and AUTO device, show the difference in performance hints (THROUGHPUT and LATENCY) with significant performance results towards the desired metric.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
