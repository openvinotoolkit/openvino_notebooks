# Text-to-Image Generation and Infinite Zoom with Stable Diffusion v2 and OpenVINO™


Stable Diffusion v2 is the next generation of Stable Diffusion model a Text-to-Image latent diffusion model created by the researchers and engineers from [Stability AI](https://stability.ai/) and [LAION](https://laion.ai/). 

General diffusion models are machine learning systems that are trained to denoise random gaussian noise step by step, to get to a sample of interest, such as an image.
Diffusion models have been shown to achieve state-of-the-art results for generating image data. But one downside of diffusion models is that the reverse denoising process is slow. In addition, these models consume a lot of memory because they operate in pixel space, which becomes unreasonably expensive when generating high-resolution images. Therefore, it is challenging to train these models and also use them for inference. OpenVINO brings capabilities to run model inference on Intel hardware and opens the door to the fantastic world of diffusion models for everyone!

In previous notebooks, we already discussed how to run [Text-to-Image generation and Image-to-Image generation using Stable Diffusion v1](../stable-diffusion-text-to-image/stable-diffusion-text-to-image.ipynb) and [controlling its generation process using ControlNet](../controlnet-stable-diffusion/controlnet-stable-diffusion.ipynb). Now, we have Stable Diffusion v2 as our latest showcase.

This notebook series demonstrates approaches to image generation using an AI method called `diffusion`:

* [Text-to-Image](./stable-diffusion-v2-text-to-image.ipynb) generation to create images from a text description as input. This notebook also contains hybrid post-training quantization of UNet model with [NNCF](https://github.com/openvinotoolkit/nncf/).

* [Text-to-Image-demo](./stable-diffusion-v2-text-to-image-demo.ipynb) is a shorter version of the original notebook for demo purposes, if would like to get started right away and run the notebook more easily.

<p align="center">
    <img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/22090501/ec454103-0d28-48e3-a18e-b55da3fab381" />
</p>

This is a demonstration in which you can type a text description and the pipeline will generate an image that reflects the context of the input text.
Step-by-step, the diffusion process will iteratively denoise latent image representation while being conditioned on the text embeddings provided by the text encoder.

The following image shows an example of the input text and the corresponding predicted image.

**Input text:** valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k

<p align="center">
    <img src="https://user-images.githubusercontent.com/1720147/229231281-065641fd-53ea-4940-8c52-b1eebfbaa7fa.png"/>
</p>

* [Text-guided Inpainting](./stable-diffusion-v2-infinite-zoom.ipynb) generation to create an image, using text description and masked image region, which should be part of the generated image.

<p align="center">
    <img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/22090501/9ac6de45-186f-4a3c-aa20-825825a337eb" />
</p>

In this demonstration Stable Diffusion v2 Inpainting model for generating sequence of images for infinite zoom video effect, extending previous images beyond its borders.

The following image shows an example of the input text and corresponding video.

**Input text:** valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k

<p align="center">
    <img src="https://user-images.githubusercontent.com/1720147/229233760-79c9425e-5691-4114-ad13-7e33f9327b52.gif"/>
</p>

* [Text-to-Image with Optimum-Intel-OpenVINO](./stable-diffusion-v2-optimum-demo.ipynb) can create images from a text description as input, using Optimum-Intel-OpenVINO. You can load optimized models from the Hugging Face Hub and create pipelines to run inference with OpenVINO Runtime without rewriting your APIs. You can run this notebook multiple times.


## For [Text-guided Inpainting](./stable-diffusion-v2-infinite-zoom.ipynb) and [Text-to-Image](./stable-diffusion-v2-text-to-image.ipynb) Notebooks

Notebooks demonstrate how to convert and run Stable Diffusion v2 models using OpenVINO.

Notebook contains the following steps:
1. Create pipeline with PyTorch models using Diffusers library.
2. Convert PyTorch models to OpenVINO IR format, using model conversion API.
3. Run Stable Diffusion v2 pipeline with OpenVINO.


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).