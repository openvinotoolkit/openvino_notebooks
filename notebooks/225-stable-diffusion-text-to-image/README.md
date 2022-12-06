# Text-to-Image Generation with Stable Diffusion

Text-to-image generation using an AI method called "diffusion" enables us to create images from a text description as input
In this notebook we demonstrate how to use a the **[Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion)** model for image generation with OpenVINO.

The complete pipeline of this demo is shown below.

![image2](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/stable_diffusion.png)

This is a demonstration in which the user can type a text description and the pipeline will generate an image that reflects the context of the input text.
Step-by-step the diffusion process will iteratively denoise latent image representation while being conditioned on the text embeddings provided by the text encoder.

The following image shows an example of the input sequence and corresponding predicted image.

![image](https://user-images.githubusercontent.com/15709723/200945747-1c584e5c-b3f2-4e43-b1c1-e35fd6edc2c3.png)

## Notebook Contents

This notebook demonstrates how to convert and run stable diffusion using OpenVINO.

Notebook contains the following steps:
1. Convert PyTorch models to ONNX format.
2. Convert ONNX models to OpenVINO IR format using Model Optimizer tool.
3. Run Stable Diffusion pipeline with OpenVINO.

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md") to install all required dependencies.