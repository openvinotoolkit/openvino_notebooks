# Line-level text detection with Surya

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/285-surya-line-level-text-detection/285-surya-line-level-text-detection.ipynb)

In this tutorial we will perform line-level text detection using [Surya](https://github.com/VikParuchuri/surya) toolkit and OpenVINO.

![line-level text detection](https://github.com/VikParuchuri/surya/blob/master/static/images/excerpt.png?raw=true)

[**image source*](https://github.com/VikParuchuri/surya)


Model used for line-level text detection based on [Segformer](https://arxiv.org/pdf/2105.15203.pdf). It has the following features:
* It is specialized for document OCR. It will likely not work on photos or other images.
* It is for printed text, not handwriting.
* The model has trained itself to ignore advertisements.
* Languages with very different character sets may not work well.

#### Table of contents:
1. Fetch test image.
1. Run PyTorch inference.
1. Convert model to OpenVINO Intermediate Representation (IR) format.
1. Run OpenVINO model.
1. Interactive inference.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
