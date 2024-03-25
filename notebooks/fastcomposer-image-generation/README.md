# Image generation with FastComposer and OpenVINO™

FastComposer uses subject embeddings extracted by an image encoder to augment the generic text conditioning in diffusion models, enabling personalized image generation based on subject images and textual instructions with only forward passes.
FastComposer generates images of multiple unseen individuals with different styles, actions, and contexts.


<img src="https://github.com/mit-han-lab/fastcomposer/blob/main/figures/multi-subject.png?raw=True" width="969">


## Notebook Contents

This notebook demonstrates how to convert and run FastComposer models using OpenVINO.

The notebook contains the following steps:
1. Convert PyTorch models to OpenVINO IR format, using model conversion API.
2. Run FastComposer pipeline with OpenVINO.

>**Note**: Please be aware that a machine with at least 32GB of RAM is necessary to run this example.


# Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
