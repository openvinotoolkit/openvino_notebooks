# Sound Generation with AudioLDM2 and OpenVINO™

[AudioLDM 2](https://huggingface.co/cvssp/audioldm2) is a latent text-to-audio diffusion model capable of generating realistic audio samples given any text input.
AudioLDM 2 was proposed in the paper [AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) by `Haohe Liu` et al.
The model takes a text prompt as input and predicts the corresponding audio. It can generate text-conditional sound effects, human speech and music.

In this tutorial we will try out the pipeline, convert the models backing it one by one and will run an interactive app with Gradio!
![](https://github.com/openvinotoolkit/openvino_notebooks/assets/76463150/c93a0f86-d9cf-4bd1-93b9-e27532170d75)


## Notebook Contents

This notebook demonstrates how to convert and run Audio LDM 2 using OpenVINO.

Notebook contains the following steps:
1. Create pipeline with PyTorch models using Diffusers library.
2. Convert PyTorch models to OpenVINO IR format using model conversion API.
3. Run Audio LDM 2 pipeline with OpenVINO.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
