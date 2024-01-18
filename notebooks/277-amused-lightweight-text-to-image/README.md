# Lightweight image generation with aMUSEd and OpenVINO™

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/277-amused-lightweight-text-to-image/277-amused-lightweight-text-to-image.ipynb)


<img src="https://huggingface.co/amused/amused-256/resolve/main/assets/collage_small.png" />

[Amused](https://huggingface.co/docs/diffusers/api/pipelines/amused) is a lightweight text to image model based off 
of the [muse](https://arxiv.org/pdf/2301.00704.pdf) architecture. Amused is particularly useful in applications that 
require a lightweight and fast model such as generating many images quickly at once.

Amused is a VQVAE token based transformer that can generate an image in fewer forward passes than many diffusion models.
 In contrast with muse, it uses the smaller text encoder CLIP-L/14 instead of t5-xxl. Due to its small parameter count 
 and few forward pass generation process, amused can generate many images quickly. This benefit is seen particularly at 
 larger batch size

## Notebook contents
The tutorial consists from following steps:

- Prerequisites
- Load and run the original pipeline
- Convert the model to OpenVINO IR
  - Convert the Text Encoder
  - Convert the U-ViT transformer
  - Convert VQ-GAN decoder (VQVAE)
- Compiling models and prepare pipeline
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).