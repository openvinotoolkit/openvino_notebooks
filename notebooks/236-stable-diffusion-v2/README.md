# Text-to-Image Generation and Infinite Zoom with Stable Diffusion v2 and OpenVINO™


Stable Diffusion V2 is the next generation of Stable Diffusion model a text-to-image latent diffusion model created by the researchers and engineers from [Stability AI](https://stability.ai/) and [LAION](https://laion.ai/). 

General diffusion models are machine learning systems that are trained to denoise random gaussian noise step by step, to get to a sample of interest, such as an image.
Diffusion models have been shown to achieve state-of-the-art results for generating image data. But one downside of diffusion models is that the reverse denoising process is slow. In addition, these models consume a lot of memory because they operate in pixel space, which becomes unreasonably expensive when generating high-resolution images. Therefore, it is challenging to train these models and also use them for inference. OpenVINO brings capabilities to run model inference on Intel hardware and opens the door to the fantastic world of diffusion models for everyone!

In previous notebooks, we already discussed how to run [Text-to-Image generation and Image-to-Image generation using Stable Diffusion v1](../225-stable-diffusion-text-to-image/225-stable-diffusion-text-to-image.ipynb) and [controlling its generation process using ControlNet](./235-controlnet-stable-diffusion/235-controlnet-stable-diffusion.ipynb). Now is the turn of Stable Diffusion V2.

This notebook demonstrates two approaches to image generation using an AI method called `diffusion`:

* `Text-to-image` generation to create images from a text description as input.
<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/228472288-be6fecb6-5ab5-411f-86dc-0e9c482c733e.png" />
</p>

This is a demonstration in which you can type a text description and the pipeline will generate an image that reflects the context of the input text.
Step-by-step, the diffusion process will iteratively denoise latent image representation while being conditioned on the text embeddings provided by the text encoder.

The following image shows an example of the input text and the corresponding predicted image.

**Input text:** valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/228775459-609e4167-fedf-4c8c-9252-f1c98a9b0e2b.png"/>
</p>


* `Text-guided Inpaining` generation to create an image, using text description and masked image region, which should be part of the generated image.
<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/228501084-60f86a71-0907-4094-a796-96350264d8b8.png" />
</p>

In this demonstration Stable Diffusion V2 Inpaining model for generating sequence of images for infinite zoom video effect, extending previous images beyond its borders.

The following image shows an example of the input text and corresponding video.

**Input text:** valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/228882108-25c1f65d-4c23-4e1d-8ba4-f6164280a3e3.gif"/>
</p>


## Notebook Contents

This notebook demonstrates how to convert and run Stable Diffusion v2 models using OpenVINO.

Notebook contains the following steps:
1. Convert PyTorch models to ONNX format.
2. Convert ONNX models to OpenVINO IR format, using Model Optimizer tool.
3. Run Stable Diffusion V2 Text-to-Image pipeline with OpenVINO.
4. Run Stable Diffusion V2 inpainting pipeline for generation infinity zoom video

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).