# Image generation with Flux.1 and OpenVINO

Flux is an AI image generation model developed by [Black Forest Labs](https://blackforestlabs.ai). It represents a significant advancement in AI-generated art, utilizing a hybrid architecture of [multimodal](https://arxiv.org/abs/2403.03206) and [parallel](https://arxiv.org/abs/2302.05442) [diffusion transformer](https://arxiv.org/abs/2212.09748) blocks and scaled to 12B parameter. The model offers state-of-the-art performance image generation with top-of-the-line prompt following, visual quality, image detail, and output diversity. More information about the model can be found in [blog post](https://blackforestlabs.ai/announcing-black-forest-labs/) and [original repo](https://github.com/black-forest-labs/flux).

<img src="https://raw.githubusercontent.com/black-forest-labs/flux/main/assets/grid.jpg" width="1024"> 

In this tutorial, we consider how to convert and optimize Flux.1 model using OpenVINO.

>**Note**: Some demonstrated models can require at least 32GB RAM for conversion and running.

### Notebook Contents

In this demonstration, you will learn how to perform text-to-image generation using Flux.1 and OpenVINO. 

Example of model work:

**Input prompt**: *a tiny Yorkshire terrier astronaut hatching from an egg on the moon*
![](https://github.com/user-attachments/assets/11733314-0b31-449c-9885-12ebf6365a58)

The tutorial consists of the following steps:

- Install prerequisites
- Collect Pytorch model pipeline
- Convert model to OpenVINO intermediate representation (IR) format 
- Compress weights using NNCF
- Prepare OpenVINO Inference pipeline
- Run Text-to-Image generation
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/flux.1-image-generation/README.md" />
