# Object segmentations with FastSAM and OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F261-fast-segment-anything%2F261-fast-segment-anything.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/261-fast-segment-anything/261-fast-segment-anything.ipynb)

The Fast Segment Anything Model (FastSAM) is a real-time CNN-based model that can segment any object within an image based on various user prompts. `Segment Anything` task is designed to make vision tasks easier by providing an efficient way to identify objects in an image. FastSAM significantly reduces computational demands while maintaining competitive performance, making it a practical choice for a variety of vision tasks.

<img src="https://user-images.githubusercontent.com/26833433/248551984-d98f0f6d-7535-45d0-b380-2e1440b52ad7.jpg" width=700>

## Notebook Contents
The tutorial consists of the following steps:

- Install and import prerequisite packages
- Download the Fast Segment Anything Model using the [Ultralytics package](https://docs.ultralytics.com/).
- Run the unconditioned segmentation mask generation pipeline
- Convert the model backing the FastSAM pipeline
- Quantize the model using NNCF
- Run interactive segmentation pipeline using OpenVINO and Gradio

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
