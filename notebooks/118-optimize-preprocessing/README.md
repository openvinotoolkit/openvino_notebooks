# Optimize preprocessing of image for the googlenet-v2 Image Classification Model with Preprocessing API in OpenVINO™

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/118-optimize-preprocessing/118-optimize-preprocessing.ipynb)

This tutorial demonstrates how the image could be transform to the data format expected by the model with Preprocessing API. Preprocessing API is an easy-to-use instrument, that enables integration of preprocessing steps into an execution graph and perform it on selected device, which can improve of device utilization. For more information about Preprocessing API, please, see this [overview](https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing.html#) and [details](https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/preprocessing-api-details.html). The tutorial uses [InceptionResNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_resnet_v2) model.


## Notebook Contents

The tutorial consists of the following steps:

* Downloading the model
* Setup preprocessing with model conversion API, loading the model and inference with original image
* Setup preprocessing with Preprocessing API, loading the model and inference with original image
* Fitting image to the model input type and inference with prepared image
* Comparing results on one picture
* Comparing performance

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).