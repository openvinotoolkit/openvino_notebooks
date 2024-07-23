# Image to Video Generation with Stable Video Diffusion

[Stable Video Diffusion (SVD)](https://stability.ai/stable-video) Image-to-Video is a diffusion model that takes in a still image as a conditioning frame, and generates a video from it. In this tutorial we consider how to convert and run Stable Video Diffusion using OpenVINO.
We will use [stable-video-diffusion-img2video-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) model as example. Additionally, to speedup video generation process we apply [AnimateLCM](https://arxiv.org/abs/2402.00769) LoRA weights and run optimization with [NNCF](https://github.com/openvinotoolkit/nncf/).

![result](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/ae8a77b2-b5c9-45c5-a103-6e46c686739f)

## Notebook Contents

This notebook demonstrates how to convert and run stable video diffusion using OpenVINO.

Notebook contains the following steps:

- Create PyTorch model pipeline using Diffusers
- Convert Stable Video Diffusion Pipeline models to OpenVINO
  - Convert Image Encoder
  - Convert U-Net
  - Convert VAE Encoder and Decoder
- Create Stable Video Diffusion Pipeline with OpenVINO
- Optimize pipeline with [NNCF](https://github.com/openvinotoolkit/nncf/)
- Compare results of original and optimized pipelines
- Interactive Demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/stable-video-diffusion/README.md" />
