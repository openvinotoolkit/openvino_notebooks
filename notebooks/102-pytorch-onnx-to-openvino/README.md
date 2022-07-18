# PyTorch to ONNX and OpenVINO™ IR Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F102-pytorch-onnx-to-openvino%2F102-pytorch-onnx-to-openvino.ipynb)

![segmentation result](https://user-images.githubusercontent.com/77325899/138498903-b616c37d-8cb3-405c-80ea-609e08470c24.png)


This notebook demonstrates how to do inference on a PyTorch semantic segmentation model, using [OpenVINO](https://github.com/openvinotoolkit/openvino).

## Notebook Contents

The notebook uses [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) to convert the open source [fastseg](https://github.com/ekzhang/fastseg/) semantic segmentation model, trained on the [CityScapes](https://www.cityscapes-dataset.com) dataset, to OpenVINO IR. It also shows how to do segmentation inference on an image, using [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html) and compares the results of the PyTorch model with the OpenVINO IR model. 

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
