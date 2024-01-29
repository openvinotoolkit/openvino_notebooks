# Text-to-image generation using PhotoMaker and OpenVINO

PhotoMaker is an efficient personalized text-to-image generation method, which mainly encodes an arbitrary number of input ID images into a stack ID embedding for preserving ID information. Such an embedding, serving as a unified ID representation, can not only encapsulate the characteristics of the same input ID comprehensively, but also accommodate the characteristics of different IDs for subsequent integration. This paves the way for more intriguing and practically valuable applications. Users can input one or a few face photos, along with a text prompt, to receive a customized photo or painting (no training required!). Additionally, this model can be adapted to any base model based on `SDXL` or used in conjunction with other `LoRA` modules.More details about PhotoMaker can be found in the [technical report](https://arxiv.org/pdf/2312.04461.pdf).

The notebook provides a simple interface that allows communication with a model using text instruction. In this demonstration user can provide input instructions and the model generates an image. 

The image below illustrates the provided generated image example.

![output](https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/88bccc4a-5789-42ca-8a68-f402c3e7c5a4)


### Notebook Contents

The tutorial consists of the following steps:

- PhotoMaker pipeline introduction
- Prerequisites
- Load original pipeline and prepare models for conversion
- Convert models to OpenVINO Intermediate representation (IR) format
- Prepare Inference pipeline
- Running Text-to-Image Generation OpenVINO
- Interactive Demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
