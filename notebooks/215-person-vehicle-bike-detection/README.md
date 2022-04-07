# Person Vehicle Bike Detection with OpenVINO

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/faseeh007/openvino_notebooks/blob/main/notebooks/215-person-vehicle-bike-detection/215-person-vehicle-bike-detection.ipynb/main?)


This demo shows how to run a MobileNetV3 Large PaddePaddle model using OpenVINO Runtime. Instead of exporting the PaddlePaddle model to ONNX and converting to Intermediate Representation (IR) format using Model Optimizer, we can now read the Paddle model directly without conversion. It also covers new features in OpenVINO 2022.1, including:

* [Preprocessing API](https://docs.openvino.ai/latest/openvino_docs_OV_Runtime_UG_Preprocessing_Overview.html)
* Directly Loading a # Person Vehicle Bike Detection-2003 Model
* [Auto-Device Plugin](https://docs.openvino.ai/latest/openvino_docs_IE_DG_supported_plugins_AUTO.html)
* [AsyncInferQueue Python API](https://docs.openvino.ai/latest/api/ie_python_api/_autosummary/openvino.runtime.AsyncInferQueue.html?highlight=asyncinferqueue#openvino.runtime.AsyncInferQueue)
* [Performance Hints](https://docs.openvino.ai/nightly/openvino_docs_OV_UG_Performance_Hints.html)
  * LATENCY Mode
  * THROUGHPUT Mode
  
## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.