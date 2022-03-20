# Introduction to Auto Device Selection in OpenVINO

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F106-auto-device%2F106-auto-device.ipynb)

|                                                                                                                             |                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| <img src="https://user-images.githubusercontent.com/36741649/128489910-316aec49-4892-46f1-9e3c-b9d3646ef278.jpg" width=300> | <img src="https://user-images.githubusercontent.com/36741649/128489933-bf215a3f-06fa-4918-8833-cb0bf9fb1cc7.jpg" width=300> |

This notebook demonstrates how to do inference with Auto Device Selection

## Notebook Contents

A basic introduction to Auto Device Selection with OpenVINO. We use the [resnet-50-tf](https://docs.openvino.ai/latest/omz_models_model_resnet_50_tf.html) model from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/). resnet-50-tf is a TensorFlow* implementation of ResNet-50 - an image classification model pre-trained on the ImageNet dataset. Originally redistributed in Saved model format, converted to frozen graph using tf.graph_util module. For details see [paper](https://arxiv.org/abs/1512.03385), [repository](https://github.com/tensorflow/models/tree/v2.2.0/official/r1/resnet).

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
