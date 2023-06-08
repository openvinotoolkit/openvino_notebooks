# Image generation with DeepFloyd IF and OpenVINO™

DeepFloyd IF is an advanced open-source text-to-image model that delivers remarkable photorealism and language comprehension. DeepFloyd IF consists of a frozen text encoder and three cascaded pixel diffusion modules: a base model that creates 64x64 px images based on text prompts and two super-resolution models, each designed to generate images with increasing resolution: 256x256 px and 1024x1024 px. All stages of the model employ a frozen text encoder, built on the T5 transformer, to derive text embeddings, which are then passed to a UNet architecture enhanced with cross-attention and attention pooling.

![deepfloyd_if_scheme](https://github.com/deep-floyd/IF/raw/develop/pics/deepfloyd_if_scheme.jpg)

## Notebook Contents

This notebook demonstrates how to convert and run DeepFloyd IF models using OpenVINO.

The notebook contains the following steps:
1. Convert PyTorch models to OpenVINO IR format, using Model Optimizer tool.
2. Run DeepFloyd IF pipeline with OpenVINO.

The result of notebook work demonstrated on the image below:
[!owl.png](https://user-images.githubusercontent.com/29454499/241643886-dfcf3c48-8d50-4730-ae28-a21595d9504f.png)

>**Note**: Please be aware that a machine with at least 32GB of RAM is necessary to run this example.


# Installation Instructions

The Jupyter notebook contains its own set of requirements installed directly within the notebook, allowing it to run independently as a standalone example.