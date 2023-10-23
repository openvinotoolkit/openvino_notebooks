# SoftVC VITS Singing Voice Conversion and OpenVINO™

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/262-softvc-voice-conversion/262-softvc-voice-conversion.ipynb)


This tutorial is based on SoftVC VITS Singing Voice Conversion project. The purpose of this project was to enable developers to have their beloved anime characters perform singing tasks. The developers' intention was to focus solely on fictional characters and avoid any involvement of real individuals, anything related to real individuals deviates from the developer's original intention.

The singing voice conversion model uses SoftVC content encoder to extract speech features from the source audio. These feature vectors are directly fed into VITS without the need for conversion to a text-based intermediate representation. As a result, the pitch and intonations of the original audio are preserved.

In this tutorial we consider how to use the base model flow to convert a singing voice to a fictional character voice with OpenVINO help.

## Notebook contents
The tutorial consists from following steps:

- Prerequisites
- Use the original model to run an inference
- Convert the original model to OpenVINO Intermediate Representation (IR) format
- Run the OpenVINO model
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).