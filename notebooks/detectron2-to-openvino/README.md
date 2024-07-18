# Convert Detection2 Models to OpenVINO™ 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fdetectron2-to-openvino%2Fdetectron2-to-openvino.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/detectron2-to-openvino/detectron2-to-openvino.ipynb)

<img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/c4dee890-6a18-4c45-8423-809653c85cb0" width=300>

[Detectron2](https://github.com/facebookresearch/detectron2) is Facebook AI Research's library that provides state-of-the-art detection and segmentation algorithms. It is the successor of [Detectron](https://github.com/facebookresearch/Detectron/) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/). It supports a number of computer vision research projects and production applications. 

In this tutorial we consider how to convert and run Detectron2 models using OpenVINO™. 

## Notebook Contents

The notebook uses `Faster R-CNN FPN x1` model and `Mask R-CNN FPN x3` pretrained on [COCO](https://cocodataset.org/#home) dataset as examples for object detection and instance segmentation respectively. It consider how to convert models to OpenVINO Intermediate Representation (IR) format and then run inference on selected inference device using OpenVINO Runtime.

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
