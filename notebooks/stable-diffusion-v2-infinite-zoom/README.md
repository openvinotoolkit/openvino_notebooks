# Infinite Zoom with Stable Diffusion v2 and OpenVINO™


Stable Diffusion v2 is the next generation of Stable Diffusion model a Text-to-Image latent diffusion model created by the researchers and engineers from [Stability AI](https://stability.ai/) and [LAION](https://laion.ai/). 

General diffusion models are machine learning systems that are trained to denoise random gaussian noise step by step, to get to a sample of interest, such as an image.
Diffusion models have been shown to achieve state-of-the-art results for generating image data. But one downside of diffusion models is that the reverse denoising process is slow. In addition, these models consume a lot of memory because they operate in pixel space, which becomes unreasonably expensive when generating high-resolution images. Therefore, it is challenging to train these models and also use them for inference. OpenVINO brings capabilities to run model inference on Intel hardware and opens the door to the fantastic world of diffusion models for everyone!

In previous notebooks, we already discussed how to run [Text-to-Image generation and Image-to-Image generation using Stable Diffusion](../stable-diffusion-text-to-image/stable-diffusion-text-to-image.ipynb) and [controlling its generation process using ControlNet](../controlnet-stable-diffusion/controlnet-stable-diffusion.ipynb). Now, we have Stable Diffusion v2 as our latest showcase.

Text-guided Inpainting generation to create an image, using text description and masked image region, which should be part of the generated image.

<p align="center">
    <img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/22090501/9ac6de45-186f-4a3c-aa20-825825a337eb" />
</p>

In this demonstration Stable Diffusion v2 Inpainting model for generating sequence of images for infinite zoom video effect, extending previous images beyond its borders.

The following image shows an example of the input text and corresponding video.

**Input text:** valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k

<p align="center">
    <img src="https://user-images.githubusercontent.com/1720147/229233760-79c9425e-5691-4114-ad13-7e33f9327b52.gif"/>
</p>


This notebook demonstrate how to convert and run Stable Diffusion v2 models using OpenVINO.

Notebook contains the following steps:
1. Create pipeline with PyTorch models using Diffusers library.
2. Convert PyTorch models to OpenVINO IR format, using model conversion API.
3. Run Stable Diffusion v2 pipeline with OpenVINO GenAI.


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/stable-diffusion-v2-infinite-zoom/README.md" />
