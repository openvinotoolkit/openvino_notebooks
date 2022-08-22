# 3D part segmentation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F224-3D-segmentation%2F224-3D-segmentation.ipynb)

![3D chair](https://user-images.githubusercontent.com/91237924/185752178-3882902c-907b-4614-b0e6-ea1de08bf3ef.png)

Point cloud is an important type of geometric data structure. Right now OpenVINO can directly consume point cloud and run inference with it.

## Notebook Contents

This notebook demonstrates how to process the point clould data and run 3D part segmentation with OpenVINO. The inputs of this task is a collection of individual data points in a three-dimensional plane with each point having a set coordinate on the X, Y, and Z axis.

In this notebook, we use the [PointNet](https://arxiv.org/abs/1612.00593) pre-trained model to detect each part of a chair and return its category.


## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.