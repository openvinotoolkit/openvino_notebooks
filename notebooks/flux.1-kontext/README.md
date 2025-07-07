# Image-to-image generation with Flux.1 Kontext and OpenVINO

FLUX.1 Kontext [dev] is a 12 billion parameter rectified flow transformer capable of editing images based on text instructions. More details about model can be found in [blog post](https://bfl.ai/announcements/flux-1-kontext-dev) and [model card](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).

<img src="https://github.com/user-attachments/assets/eaa37f44-6e9f-4c22-b5ed-eb1816e98cd0" width="1024"> 

In this tutorial, we consider how to convert and optimize Flux.1 Kontext model using OpenVINO.

>**Note**: Some demonstrated models can require at least 32GB RAM for conversion and running.

### Notebook Contents

In this demonstration, you will learn how to perform image-to-image generation using Flux.1 Kontext and OpenVINO. 

Example of model work:

**Input prompt**: *a tiny Yorkshire terrier astronaut hatching from an egg on the moon*
![](https://github.com/user-attachments/assets/11733314-0b31-449c-9885-12ebf6365a58)

The tutorial consists of the following steps:

- Install prerequisites
- Collect Pytorch model pipeline
- Convert model to OpenVINO intermediate representation (IR) format 
- Compress weights using NNCF
- Prepare OpenVINO Inference pipeline
- Run Image-to-Image generation
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/flux.1-kontext/README.md" />
