# Working with GPUs in OpenVINO™

This notebook shows how to do inference with Graphic Processing Units (GPUs). To learn more about GPUs in OpenVINO, refer to the [GPU Device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) section in the docs.

## Notebook Contents

This notebook provides a high-level overview of working with Intel GPUs in OpenVINO. It shows how to use Query Device to list system GPUs and check their properties, and it explains some of the key properties. It shows how to compile a model on GPU with performance hints and how to use multiple GPUs using MULTI or CUMULATIVE_THROUGHPUT.

The notebook also presents example commands for benchmark_app that can be run to compare GPU performance in different configurations. It also provides the code for a basic end-to-end application that compiles a model on GPU and uses it to run inference.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/gpu-device/README.md" />
