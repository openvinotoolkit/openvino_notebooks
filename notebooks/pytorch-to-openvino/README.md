# PyTorch to OpenVINO™ IR Tutorial

This tutorial demonstrates how to convert PyTorch models to OpenVINO Intermediate Representation (IR) format.

## Notebook Contents

* [pytorch-to-openvino](./pytorch-to-openvino.ipynb) shows how to convert the Pytorch model in formats `torch.nn.Module` and `torch.jit.ScriptModule` into OpenVINO Intermediate Representation. The tutorial uses [RegNetY_800MF](https://arxiv.org/abs/2003.13678) model from [torchvision](https://pytorch.org/vision/stable/index.html) pre-trained on [ImageNet](https://www.image-net.org/) dataset to demonstrate how to convert PyTorch models to OpenVINO Intermediate Representation using Model Converter. It also shows how to do classification inference on an image, using [OpenVINO Runtime](https://docs.openvino.ai/2024/openvino-workflow/running-inference.html) and compares the results of the PyTorch model with the OpenVINO IR model.

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pytorch-to-openvino/pytorch-to-openvino.ipynb)
![classification_result](https://user-images.githubusercontent.com/29454499/250586825-2a4a74a6-e091-4e47-8f29-59a72fe4975f.png)


* [pytorch-onnx-to-openvino](./pytorch-onnx-to-openvino.ipynb) shows how to convert the PyTorch model to OpenVINO IR with the intermediate step of exporting PyTorch model to ONNX format.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fpytorch-to-openvino%2Fpytorch-onnx-to-openvino.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pytorch-to-openvino/pytorch-onnx-to-openvino.ipynb)
![segmentation result](https://user-images.githubusercontent.com/29454499/203723317-1716e3ca-b390-47e1-bb98-07b4d8d097a0.png)

The notebook uses OpenVINO Model Converter (OVC) to convert the open-source Lite-RASPP semantic segmentation model with a MobileNet V3 Large backbone from [torchvision](https://pytorch.org/vision/main/models/lraspp.html), trained on [COCO](https://cocodataset.org) dataset images using 20 categories that are present in the [Pascal VOC](https://paperswithcode.com/dataset/pascal-voc) dataset, to OpenVINO IR. It also shows how to do segmentation inference on an image, using [OpenVINO Runtime](https://docs.openvino.ai/2024/openvino-workflow/running-inference.html) and compares the results of the PyTorch model with the OpenVINO IR model.


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/pytorch-to-openvino/README.md" />
