# Image generation with universal control using Flex.2 and OpenVINO

Flex.2 is flexible text-to-image diffusion model based on Flux model architecture with built in support inpainting and universal control - model accepts pose, line, and depth inputs.

More details about model can be found in [model card](https://huggingface.co/ostris/Flex.2-preview).

In this tutorial we consider how to convert and optimize Flex.2 model using OpenVINO.

>**Note**: Some demonstrated models can require at least 32GB RAM for conversion and running.

<img src="https://raw.githubusercontent.com/black-forest-labs/flux/main/assets/grid.jpg" width="1024"> 

### Notebook Contents

In this demonstration, you will learn how to perform text-to-image generation using Flex.2 and OpenVINO. 

Example of model work:

**Input prompt**: *a tiny Yorkshire terrier astronaut hatching from an egg on the moon*
![](https://github.com/user-attachments/assets/11733314-0b31-449c-9885-12ebf6365a58)

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