# Image Generation with Stable Diffusion


This notebook demonstrates how to use a **[Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion)** model for image generation with OpenVINO.
It considers two approaches of image generation using an AI method called `diffusion`:

* `Text-to-image` generation to create images from a text description as input.
* `Text-guided Image-to-Image` generation to create an image, using text description and initial image semantic.

The complete pipeline of this demo is shown below.

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/216378932-7a9be39f-cc86-43e4-b072-66372a35d6bd.png"/>
</p>


This is a demonstration in which you can type a text description (and provide input image in case of Image-to-Image generation) and the pipeline will generate an image that reflects the context of the input text.
Step-by-step, the diffusion process will iteratively denoise latent image representation while being conditioned on the text embeddings provided by the text encoder.

The following image shows an example of the input sequence and corresponding predicted image.

**Input text:** cyberpunk cityscape like Tokyo, New York with tall buildings at dusk golden hour cinematic lighting, epic composition. A golden daylight, hyper-realistic environment. Hyper and intricate detail, photo-realistic. Cinematic and volumetric light. Epic concept art. Octane render and Unreal Engine, trending on artstation

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/216524089-ed671fc7-a78b-42bf-aa96-9f7c791a9419.png"/>
</p>

## Notebook Contents

This notebook demonstrates how to convert and run stable diffusion using OpenVINO.

Notebook contains the following steps:
1. Convert PyTorch models to ONNX format.
2. Convert ONNX models to OpenVINO IR format, using Model Optimizer tool.
3. Run Stable Diffusion pipeline with OpenVINO.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
