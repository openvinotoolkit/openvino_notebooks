# Image editng with InstructPix2Pix

AI image editing models are traditionally focused a single editing task such as style transfer or translation between image domains. [InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix/) propose a nowel method for editing images from human instructions: given an input image and a written instruction that tells the model what to do, the model follows these instructions to edit the image.

In this notebook we demonstrate how to use a the **[InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)** model for image editing with OpenVINO.

The complete pipeline of this demo is shown below.

![diagram](https://user-images.githubusercontent.com/29454499/214895365-3063ac11-0486-4d9b-9e25-8f469aba5e5d.png)

This is a demonstration in which the user can type a text instruction and provide input image the pipeline will generate an image that reflects the context of the input text.
Step-by-step the diffusion process will iteratively denoise latent image representation while being conditioned on the text embeddings provided by the text encoder and original image encoded by variatioanl auto encoder.

The following image shows an example of the input image and prompt and corresponding edited image.

![image](https://user-images.githubusercontent.com/29454499/214905933-eda1b88d-ccc5-45a1-bc12-bb5e382811fb.png)

## Notebook Contents

This notebook demonstrates how to convert and run stable diffusion using OpenVINO.

Notebook contains the following steps:
1. Convert PyTorch models to ONNX format.
2. Convert ONNX models to OpenVINO IR format using Model Optimizer tool.
3. Run InstructPix2Pix pipeline with OpenVINO.

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md) to install all required dependencies.
