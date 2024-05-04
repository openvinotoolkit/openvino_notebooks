# Object detection and masking from prompts with GroundedSAM (GroundingDINO + SAM) and OpenVINO

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/288-grounded-segment-anything/288-grounded-segment-anything.ipynb)

In this notebook, we provide the OpenVINO™ optimization for the combination of GroundingDINO + SAM = [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) on Intel® platforms. 

GroundedSAM aims to detect and segment anything with text inputs. GroundingDINO is a language-guided query selection module to enhance object detection using input text. It selects relevant features from image and text inputs and returns predicted boxes with detections. The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. We use box predictions from GroundingDINO to mask the original image.

More details about the model can be found in the [paper](https://arxiv.org/abs/2401.14159), and the official [repository](https://github.com/IDEA-Research/Grounded-Segment-Anything).

In this tutorial, we will explore how to convert and run GroundedSAM using OpenVINO.

![image](https://github.com/openvinotoolkit/openvino_notebooks/assets/5703039/3c19063a-c60a-4d5d-b534-e1305a854180)

## Notebook Contents
- Download checkpoints and load PyTorch model
- Convert GroundingDINO to OpenVINO IR format
- Run OpenVINO optimized GroundingDINO
- Convert SAM to OpenVINO IR
- Combine GroundingDINO + SAM (GroundedSAM)
- Interactive GroundedSAM

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
