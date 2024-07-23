# Image generation with StableCascade and OpenVINO

<img src="https://huggingface.co/stabilityai/stable-cascade/resolve/main/figures/collage_1.jpg" />

[Stable Cascade](https://huggingface.co/stabilityai/stable-cascade) is built upon the [Würstchen](https://openreview.net/forum?id=gU58d5QeGv) architecture and its main difference to other models like Stable Diffusion is that it is working at a much smaller latent space. Why is this important? The smaller the latent space, the faster you can run inference and the cheaper the training becomes. How small is the latent space? Stable Diffusion uses a compression factor of 8, resulting in a 1024x1024 image being encoded to 128x128. Stable Cascade achieves a compression factor of 42, meaning that it is possible to encode a 1024x1024 image to 24x24, while maintaining crisp reconstructions. The text-conditional model is then trained in the highly compressed latent space.

The notebook provides a simple interface that allows communication with a model using text instruction. In this demonstration user can provide input instructions and the model generates an image. An additional part demonstrates how to use weights compression with [NNCF](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html#compress-model-weights) to speed up pipeline and reduce memory consumption.

>**Note**: This demonstration can require about 50GB RAM for conversion and running.

## Notebook contents
This tutorial consists of the following steps:
- Prerequisites
- Load the original model
    - Infer the original model
- Convert the model to OpenVINO IR
    - Prior pipeline
    - Decoder pipeline
- Compiling models
- Building the pipeline
- Inference
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/stable-cascade-image-generation/README.md" />
