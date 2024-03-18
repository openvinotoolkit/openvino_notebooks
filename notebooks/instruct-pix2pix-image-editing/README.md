# Image editing with InstructPix2Pix

AI image editing models are traditionally focused on a single editing task such as style transfer or translation between image domains. [InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix/) proposes a novel method for editing images using human instructions given an input image and a written text that tells the model what to do. The model follows these text-based instructions to edit the image.

This notebook demonstrates how to use the **[InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)** model for image editing with OpenVINO.

The complete pipeline of this demo is shown below.

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/214895365-3063ac11-0486-4d9b-9e25-8f469aba5e5d.png"/>
</p>

This is a demonstration in which you can type text-based instructions and provide an input image to the pipeline that will generate a new image, that reflects the context of the input text.
Step-by-step the diffusion process will iteratively denoise the latent image representation while being conditioned on the text embeddings, provided by the text encoder and an original image encoded by a variational autoencoder.

The following image shows an example of the input image with text-based prompt and the corresponding edited image.

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/219943222-d46a2e2d-d348-4259-8431-37cf14727eda.png"/>
</p>

## Notebook Contents

This notebook demonstrates how to convert and run stable diffusion using OpenVINO.

Notebook contains the following steps:
1. Convert PyTorch models to OpenVINO IR format, using Model Conversion API.
2. Run InstructPix2Pix pipeline with OpenVINO.
3. Optimize InstructPix2Pix pipeline with [NNCF](https://github.com/openvinotoolkit/nncf/) quantization.
4. Compare results of original and optimized pipelines.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
