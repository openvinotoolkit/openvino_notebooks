# PyTorch to OpenVINO™ IR Tutorial

This tutorial demonstrates how to convert PyTorch models to OpenVINO Intermediate Representation (IR) format.

## Notebook Contents

The notebook shows how to convert the Pytorch model in formats `torch.nn.Module` and `torch.jit.ScriptModule` into OpenVINO Intermediate Representation. The tutorial uses [RegNetY_800MF](https://arxiv.org/abs/2003.13678) model from [torchvision](https://pytorch.org/vision/stable/index.html) pre-trained on [ImageNet](https://www.image-net.org/) dataset to demonstrate how to convert PyTorch models to OpenVINO Intermediate Representation using Model Converter. It also shows how to do classification inference on an image, using [OpenVINO Runtime](https://docs.openvino.ai/2024/openvino-workflow/running-inference.html) and compares the results of the PyTorch model with the OpenVINO IR model.

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pytorch-to-openvino/pytorch-to-openvino.ipynb)
![classification_result](https://user-images.githubusercontent.com/29454499/250586825-2a4a74a6-e091-4e47-8f29-59a72fe4975f.png)


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/pytorch-to-openvino/README.md" />
