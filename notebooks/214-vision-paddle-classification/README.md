# PaddlePaddle Image Classification with OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F214-vision-paddle-classification%2F214-vision-paddle-classification.ipynb)


This demo shows how to run a MobileNetV3 Large PaddePaddle model, using OpenVINO Runtime. Instead of exporting the PaddlePaddle model to ONNX and converting to OpenVINO Intermediate Representation (OpenVINO IR) format by using Model Optimizer, you can now read the Paddle model directly without conversion. 

## Notebook Contents

This tutorial also covers new features in OpenVINO 2022.1, including:

* [Preprocessing API](https://docs.openvino.ai/latest/openvino_docs_OV_Runtime_UG_Preprocessing_Overview.html)
* Directly Loading a PaddlePaddle Model
* [Auto-Device Plugin](https://docs.openvino.ai/latest/openvino_docs_IE_DG_supported_plugins_AUTO.html)
* [AsyncInferQueue Python API](https://docs.openvino.ai/latest/api/ie_python_api/_autosummary/openvino.runtime.AsyncInferQueue.html?highlight=asyncinferqueue#openvino.runtime.AsyncInferQueue)
* [Performance Hints](https://docs.openvino.ai/nightly/openvino_docs_OV_UG_Performance_Hints.html)
  * LATENCY Mode
  * THROUGHPUT Mode
  
## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
