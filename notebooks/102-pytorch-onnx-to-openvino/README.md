# PyTorch to ONNX and OpenVINO™ IR Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F102-pytorch-onnx-to-openvino%2F102-pytorch-onnx-to-openvino.ipynb)

![segmentation result](https://user-images.githubusercontent.com/77325899/138498903-b616c37d-8cb3-405c-80ea-609e08470c24.png)


This notebook demonstrates how to do inference on a PyTorch semantic segmentation model, using [OpenVINO](https://github.com/openvinotoolkit/openvino).

## Notebook Contents

The notebook uses [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) to convert the open source a DeepLabV3 model with a MobileNetV3-Large backbone semantic segmentation model from [torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html#torchvision.models.segmentation.deeplabv3_mobilenet_v3_large), trained on [COCO](https://cocodataset.org) dataset images using 20 categories that are present in the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) dataset, to OpenVINO IR. It also shows how to do segmentation inference on an image, using [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html) and compares the results of the PyTorch model with the OpenVINO IR model. 

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
