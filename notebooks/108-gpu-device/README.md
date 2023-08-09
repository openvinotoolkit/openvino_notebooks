# Working with GPUs in OpenVINO™

This notebook shows how to do inference with Graphic Processing Units (GPUs). To learn more about GPUs in OpenVINO, refer to the [GPU Device](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_supported_plugins_GPU.html) section in the docs.

## Notebook Contents

This notebook provides a high-level overview of working with Intel GPUs in OpenVINO. It shows how to use Query Device to list system GPUs and check their properties, and it explains some of the key properties. It shows how to compile a model on GPU with performance hints and how to use multiple GPUs using MULTI or CUMULATIVE_THROUGHPUT.

The notebook also presents example commands for benchmark_app that can be run to compare GPU performance in different configurations. It also provides the code for a basic end-to-end application that compiles a model on GPU and uses it to run inference.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).