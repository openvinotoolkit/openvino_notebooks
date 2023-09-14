# Convert TensorFlow Object Detection and Instance Segmentation Models to OpenVINO™ 

This tutorial explains how to convert [TensorFlow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) models to OpenVINO IR.

## Object Detection

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F120-tensorflow-object-detection-to-openvino%2F120-tensorflow-object-detection-to-openvino.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/120-tensorflow-object-detection-to-openvino/120-tensorflow-object-detection-to-openvino.ipynb)

<img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/41733560/f9b59be1-1d2f-4e13-9678-67205be78841" width=300>

### Notebook Contents

The notebook shows how to convert the [Faster R-CNN with Resnet-50 V1 Object Detection model](https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1) and then detect objects in an image using OpenVINO Runtime.

### Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

## Instance segmentation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F120-tensorflow-object-detection-to-openvino%2F120-tensorflow-instance-segmentation-to-openvino.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/120-tensorflow-object-detection-to-openvino/120-tensorflow-instance-segmentation-to-openvino.ipynb)

<img src="https://github.com/itrushkin/openvino_notebooks/assets/76161256/b0acc7bb-92c7-4c83-9484-84a8e1b798b7" width=300/>


### Notebook Contents

The notebook shows how to convert the [Mask R-CNN with Inception ResNet V2 Instance Segmentation model](https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1) and then segment instances in an image using OpenVINO Runtime.

### Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
