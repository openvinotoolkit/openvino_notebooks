# PyTorch to OpenVINO™ IR Tutorial

This tutorial demonstrates how to convert PyTorch models to OpenVINO Intermediate Representation (IR) format.

## Notebook Contents

* [102-pytorch-to-openvino](./102-pytorch-to-openvino.ipynb) shows how to convert the Pytorch model in formats `torch.nn.Module` and `torch.jit.ScriptModule` into OpenVINO Intermediate Representation. The tutorial uses [RegNetY_800MF](https://arxiv.org/abs/2003.13678) model from [torchvision](https://pytorch.org/vision/stable/index.html) pre-trained on [ImageNet](https://www.image-net.org/) dataset to demonstrate how to convert PyTorch models to OpenVINO Intermediate Representation using Model Converter. It also shows how to do classification inference on an image, using [OpenVINO Runtime](https://docs.openvino.ai/nightly/openvino_docs_OV_UG_OV_Runtime_User_Guide.html) and compares the results of the PyTorch model with the OpenVINO IR model.

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/102-pytorch-to-openvino/102-pytorch-to-openvino.ipynb)
![classification_result](https://user-images.githubusercontent.com/29454499/250586825-2a4a74a6-e091-4e47-8f29-59a72fe4975f.png)


* [102-pytorch-onnx-to-openvino](./102-pytorch-onnx-to-openvino.ipynb) shows how to convert the PyTorch model to OpenVINO IR with the intermediate step of exporting PyTorch model to ONNX format.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F102-pytorch-onnx-to-openvino%2F102-pytorch-onnx-to-openvino.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/102-pytorch-to-openvino/102-pytorch-onnx-to-openvino.ipynb)
![segmentation result](https://user-images.githubusercontent.com/29454499/203723317-1716e3ca-b390-47e1-bb98-07b4d8d097a0.png)

The notebook uses OpenVINO Model Converter (OVC) to convert the open-source Lite-RASPP semantic segmentation model with a MobileNet V3 Large backbone from [torchvision](https://pytorch.org/vision/main/models/lraspp.html), trained on [COCO](https://cocodataset.org) dataset images using 20 categories that are present in the [Pascal VOC](https://paperswithcode.com/dataset/pascal-voc) dataset, to OpenVINO IR. It also shows how to do segmentation inference on an image, using [OpenVINO Runtime](https://docs.openvino.ai/nightly/openvino_docs_OV_UG_OV_Runtime_User_Guide.html) and compares the results of the PyTorch model with the OpenVINO IR model.
 

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).