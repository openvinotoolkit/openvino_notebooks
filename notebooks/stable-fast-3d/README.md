# Stable Fast 3D Mesh Reconstruction and OpenVINO

<div class="alert alert-block alert-danger"> <b>Important note:</b> This notebook requires python >= 3.9. Please make sure that your environment fulfill to this requirement before running it </div>

[Stable Fast 3D (SF3D)](https://huggingface.co/stabilityai/stable-fast-3d) is a large reconstruction model based on [TripoSR](https://huggingface.co/spaces/stabilityai/TripoSR), which takes in a single image of an object and generates a textured UV-unwrapped 3D mesh asset.

You can find [the source code on GitHub](https://github.com/Stability-AI/stable-fast-3d) and read the paper [SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and Illumination Disentanglement](https://arxiv.org/abs/2408.00653).

![Teaser Video](https://github.com/Stability-AI/stable-fast-3d/blob/main/demo_files/teaser.gif?raw=true)


## Notebook contents
The tutorial consists from following steps:

- Prerequisites
- Get the original model
- Convert the original model to OpenVINO Intermediate Representation (IR) format
- Compiling models and prepare pipeline
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/stable-fast-3d/README.md" />
