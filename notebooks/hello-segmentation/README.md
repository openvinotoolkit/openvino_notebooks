# Introduction to Segmentation in OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fhello-segmentation%2Fhello-segmentation.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/hello-segmentation/hello-segmentation.ipynb)

|                                                                                                                             |                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| <img src="https://user-images.githubusercontent.com/36741649/127848003-9e45c8da-2e43-48ac-803f-9f51a8e9ea89.jpg" width=300> | <img src="https://user-images.githubusercontent.com/36741649/127847882-6305d483-f2ce-4c2f-a3b5-8573d1522d15.png" width=300> |

This notebook demonstrates how to do inference with segmentation model.

## Notebook Contents

A very basic introduction to segmentation with OpenVINO. This notebook uses the [`road-segmentation-adas-0001`](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/road-segmentation-adas-0001/README.md) model from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/) and an input image downloaded from [Mapillary Vistas](https://www.mapillary.com/dataset/vistas). ADAS stands for Advanced Driver Assistance Services. The model recognizes four classes: background, road, curb and mark.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/hello-segmentation/README.md" />
