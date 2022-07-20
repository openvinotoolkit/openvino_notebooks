# Background Removal Demo

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F205-vision-background-removal%2F205-vision-background-removal.ipynb)

This demo notebook shows image segmentation and removing/adding background with [U^2-Net](https://github.com/xuebinqin/U-2-Net) and OpenVINO™.

![Image segmentation with U^2-Net and OpenVINO](https://user-images.githubusercontent.com/77325899/116818525-1ca00980-ab6c-11eb-83b4-d42fa7d6d94a.png)
![Background removal with U^2-Net and OpenVINO](https://user-images.githubusercontent.com/77325899/116818585-74d70b80-ab6c-11eb-9bad-1ddf1b5ea5fe.png)

## Notebook Contents

* Importing Pytorch library and loading U^2-Net model.
* Converting PyTorch U^2-Net model to ONNX and OpenVINO IR.
* Loading and preprocessing input image.
* Doing inference on OpenVINO IR model.
* Visualizing results.

## U^2-Net source

``` markdown
@InProceedings{Qin_2020_PR,
title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
journal = {Pattern Recognition},
volume = {106},
pages = {107404},
year = {2020}
}
```

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
