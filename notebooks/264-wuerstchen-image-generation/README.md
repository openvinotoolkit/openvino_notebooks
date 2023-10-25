# Image generation with Würstchen and OpenVINO
| Anthropomorphic cat dressed as a firefighter |
| --- |
| <img src="https://github.com/itrushkin/openvino_notebooks/assets/76161256/6917c558-d74c-4cc9-b81a-679ce0a299ee" /> |

Würstchen is a diffusion model, whose text-conditional model works in a highly compressed latent space of images. Why is this important? Compressing data can reduce computational costs for both training and inference by magnitudes. Training on 1024x1024 images, is way more expensive than training at 32x32. Usually, other works make use of a relatively small compression, in the range of 4x - 8x spatial compression. Würstchen takes this to an extreme. Through its novel design, authors achieve a 42x spatial compression. This was unseen before because common methods fail to faithfully reconstruct detailed images after 16x spatial compression. Würstchen employs a two-stage compression, what we call Stage A and Stage B. Stage A is a VQGAN, and Stage B is a Diffusion Autoencoder (more details can be found in the [paper](https://arxiv.org/abs/2306.00637)). A third model, Stage C, is learned in that highly compressed latent space. This training requires fractions of the compute used for current top-performing models, allowing also cheaper and faster inference.

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
