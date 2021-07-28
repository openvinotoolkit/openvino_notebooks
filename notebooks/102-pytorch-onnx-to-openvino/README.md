# PyTorch to ONNX and OpenVINO IR Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F102-pytorch-onnx-to-openvino%2F102-pytorch-onnx-to-openvino.ipynb)

![segmentation](https://user-images.githubusercontent.com/36741649/127172810-6d5376bf-3613-4b4d-a036-21123bb35857.jpg)

This notebook demonstrates how to perform inference on a PyTorch semantic segmentation model using [OpenVINO](https://github.com/openvinotoolkit/openvino).

## Notebook Contents

The notebook uses OpenVINO [Model Optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) to convert the open source [fastseg](https://github.com/ekzhang/fastseg/) semantic segmentation model, trained on the [CityScapes](https://www.cityscapes-dataset.com) dataset, to OpenVINO IR. It also shows how to perform segmentation inference on an image using OpenVINO [Inference Engine](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html) and compares the results of the PyTorch model with the IR model. 

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
