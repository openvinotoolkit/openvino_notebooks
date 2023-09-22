# TorchVision Zoo with OpenVINO™

The `torchvision.models` subpackage contains definitions of models for addressing different tasks, including: image 
classification, pixelwise semantic segmentation, object detection, instance segmentation, person keypoint detection, 
video classification, and optical flow.

# Contents:
Throughout this notebook we will learn how to convert these pretrained model to OpenVINO. Here are two examples:

1. ConvNext classification model. In these example will be also demonstrating how to convert a 
`torchvision.transforms` object into OpenVINO preprocessing.
2. LRASPP MobileNetV3 semantic segmentation model. 

# Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).