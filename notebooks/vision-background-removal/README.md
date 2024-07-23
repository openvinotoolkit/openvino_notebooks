# Background Removal Demo

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fvision-background-removal%2Fvision-background-removal.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/vision-background-removal/vision-background-removal.ipynb)

This demo notebook shows image segmentation and removing/adding background with [U^2-Net](https://github.com/xuebinqin/U-2-Net) and OpenVINO™.

![Image segmentation with U^2-Net and OpenVINO](https://user-images.githubusercontent.com/77325899/116818525-1ca00980-ab6c-11eb-83b4-d42fa7d6d94a.png)
![Background removal with U^2-Net and OpenVINO](https://user-images.githubusercontent.com/77325899/116818585-74d70b80-ab6c-11eb-9bad-1ddf1b5ea5fe.png)

## Notebook Contents

* Importing Pytorch library and loading U^2-Net model.
* Converting PyTorch U^2-Net model to OpenVINO IR format.
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

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
