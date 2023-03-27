# PyTorch to ONNX and OpenVINO™ IR Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F102-pytorch-onnx-to-openvino%2F102-pytorch-onnx-to-openvino.ipynb)

![segmentation result](https://user-images.githubusercontent.com/29454499/203723317-1716e3ca-b390-47e1-bb98-07b4d8d097a0.png)


This notebook demonstrates how to do inference on a PyTorch semantic segmentation model, using [OpenVINO](https://github.com/openvinotoolkit/openvino).

## Notebook Contents

The notebook uses [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) to convert the open-source Lite-RASPP semantic segmentation model with a MobileNet V3 Large backbone from [torchvision](https://pytorch.org/vision/main/models/lraspp.html), trained on [COCO](https://cocodataset.org) dataset images using 20 categories that are present in the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) dataset, to OpenVINO IR. It also shows how to do segmentation inference on an image, using [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html) and compares the results of the PyTorch model with the OpenVINO IR model. 

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).