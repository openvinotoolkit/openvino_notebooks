# Text-to-Music generation using Riffusion and OpenVINO

Riffusion is a latent text-to-image diffusion model capable of generating spectrogram images given any text input. These spectrograms can be converted into audio clips.
General diffusion models are machine learning systems that are trained to denoise random gaussian noise step by step, to get to a sample of interest, such as an image.
Diffusion models have shown to achieve state-of-the-art results for generating image data. But one downside of diffusion models is that the reverse denoising process is slow. In addition, these models consume a lot of memory because they operate in pixel space, which becomes unreasonably expensive when generating high-resolution images. Therefore, it is challenging to train these models and also use them for inference. OpenVINO brings capabilities to run model inference on Intel hardware and opens the door to the fantastic world of diffusion models for everyone!

In this tutorial, we consider how to run an text-to-music generation pipeline using Riffusion and OpenVINO. We will use a pre-trained model from the [Diffusers](https://huggingface.co/docs/diffusers/index) library. To simplify the user experience, the [Hugging Face Optimum Intel](https://huggingface.co/docs/optimum/intel/index) library is used to convert the models to OpenVINO™ IR format.

The complete pipeline of this demo is shown below.

![riffusion_pipeline.png](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/3de12c6b-23ef-4953-aeda-5785108990b9)

## Notebook Contents

This notebook demonstrates how to convert and run riffusion using OpenVINO.

The tutorial consists of the following steps:

- Install prerequisites
- Download and convert the model from a public source using the [OpenVINO integration with Hugging Face Optimum](https://huggingface.co/blog/openvino).
- Create an text-to-music inference pipeline
- Run inference pipeline


This notebook provides interactive interface, where user can insert own musical input prompt and model will generate spectrogram image and sound guided by provided input. The result of demo work illustrated on image below.

![demo_riffusion.png](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d14743bd-d5de-4527-9000-f6090d86e9ac)

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
