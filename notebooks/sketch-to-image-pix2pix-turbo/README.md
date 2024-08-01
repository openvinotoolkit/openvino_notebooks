# One Step Sketch to Image translation with pix2pix-turbo and OpenVINO

Diffusion models achieve remarkable results in image generation. They are able synthesize high-quality images guided by user instructions. In the same time, majority of diffusion-based image generation approaches are time-consuming due to the iterative denoising process.Pix2Pix-turbo model was proposed in [One-Step Image Translation with Text-to-Image Models paper](https://arxiv.org/abs/2403.12036) for addressing slowness of diffusion process in image-to-image translation task. It is based on [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo), a fast generative text-to-image model that can synthesize photorealistic images from a text prompt in a single network evaluation. Using only single inference, pix2pix-turbo achieves comparable by quality results with recent works such as ControlNet for Sketch2Photo and Edge2Image for 50 steps.

![](https://github.com/GaParmar/img2img-turbo/raw/main/assets/gen_variations.jpg)

In this tutorial you will learn how to turn sketches into images using [Pix2Pix-Turbo](https://github.com/GaParmar/img2img-turbo) and OpenVINO.

## Notebook contents
The tutorial consists from following steps:

- Prerequisites
- Load PyTorch Model
- Convert the model to OpenVINO IR
- Select Inference Device
- Compile OpenVINO Model
- Run Model Inference
- Launch Interactive Demo

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/sketch-to-image-pix2pix-turbo/README.md" />
