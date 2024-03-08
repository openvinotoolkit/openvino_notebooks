# PaddlePaddle to OpenVINO™ IR Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2F103-paddle-to-openvino%2F103-paddle-to-openvino-classification.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/103-paddle-to-openvino/103-paddle-to-openvino-classification.ipynb)

![PaddlePaddle Classification](https://user-images.githubusercontent.com/77325899/127503530-72c8ce57-ef6f-40a7-808a-d7bdef909d11.png)

This notebook shows how to convert [PaddlePaddle](https://www.paddlepaddle.org.cn) models to OpenVINO IR.

## Notebook Contents

The notebook uses [model conversion API](https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html) to convert a MobileNet V3 [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) model, pre-trained on the [ImageNet](https://www.image-net.org) dataset, to OpenVINO IR. It also shows how to perform classification inference on an image, using [OpenVINO Runtime](https://docs.openvino.ai/2024/openvino-workflow/running-inference.html) and compares the results of the PaddlePaddle model with the OpenVINO IR model.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
