# Part Segmentation of 3D Point Clouds with OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2F3D-segmentation-point-clouds%2F3D-segmentation-point-clouds.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/3D-segmentation-point-clouds/3D-segmentation-point-clouds.ipynb)

<p align="center">
    <img src="https://user-images.githubusercontent.com/91237924/185752178-3882902c-907b-4614-b0e6-ea1de08bf3ef.png"/>
</p>

Point clouds are an important type of geometric data structure. OpenVINO can directly consume point cloud data and perform inference with it.

## Notebook Contents

This notebook demonstrates how to process [point cloud](https://en.wikipedia.org/wiki/Point_cloud) data and run 3D Part Segmentation with OpenVINO. The inputs of this task are a collection of individual data points in a three-dimensional plane with each point having a set coordinates on the X, Y, and Z axes.

This notebook uses a pre-trained [PointNet](https://arxiv.org/abs/1612.00593) model to detect each part of a chair and return its category.


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/3D-segmentation-point-clouds/README.md" />
