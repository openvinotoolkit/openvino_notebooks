# Image generation with universal control using Flex.2 and OpenVINO

<div class="alert alert-block alert-danger"> <b>Important note:</b> This notebook requires python >= 3.11. Please make sure that your environment fulfill to this requirement before running it </div>

Flex.2 is flexible text-to-image diffusion model based on Flux model architecture with built in support inpainting and universal control - model accepts pose, line, and depth inputs.

<img src="https://github.com/user-attachments/assets/6a9ab66a-387a-4538-8625-2bb3a16072b5" width="1024"> 

More details about model can be found in [model card](https://huggingface.co/ostris/Flex.2-preview).

In this tutorial we consider how to convert and optimize Flex.2 model using OpenVINO.

>**Note**: Some demonstrated models can require at least 32GB RAM for conversion and running.

### Notebook Contents

In this demonstration, you will learn how to perform text-to-image generation using Flex.2 and OpenVINO. 

Example of model work:

![](https://github.com/user-attachments/assets/140685b7-2c5d-4cef-86fb-33df0849ec1a)

The tutorial consists of the following steps:

- Install prerequisites
- Collect Pytorch model pipeline
- Convert model to OpenVINO intermediate representation (IR) format 
- Compress weights using NNCF
- Prepare OpenVINO Inference pipeline
- Run Image generation
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/flex.2-image-generation/README.md" />
