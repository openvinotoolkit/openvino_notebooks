# Subject-driven image generation and editing using BLIP Diffusion and OpenVINO
| Task | Inference result demo |
| --- | --- |
| Zero-shot subject-driven generation | ![image](https://github.com/itrushkin/openvino_notebooks/assets/76161256/fcc48429-2a79-4ada-b04a-c02326667be0) |
| Controlled subject-driven generation (Canny-edge) | ![image](https://github.com/itrushkin/openvino_notebooks/assets/76161256/2469b60e-c22b-4d29-bf27-a88b736d7129) |
| Controlled subject-driven generation (Scribble) | ![image](https://github.com/itrushkin/openvino_notebooks/assets/76161256/350617aa-b5dc-4baa-9e59-47a360602c04) |

[BLIP-Diffusion](https://arxiv.org/abs/2305.14720) is a text-to-image diffusion model with built-in support for multimodal subject-and-text condition. BLIP-Diffusion enables zero-shot subject-driven generation, and efficient fine-tuning for customized subjects with up to 20x speedup. In addition, BLIP-Diffusion can be flexibly combined with ControlNet and prompt-to-prompt to enable novel subject-driven generation and editing applications.

## Notebook contents
The tutorial consists of the following steps:

- Prerequisites
- Load the model
- Infer the original model
    - Zero-Shot subject-driven generation
    - Controlled subject-driven generation (Canny-edge)
    - Controlled subject-driven generation (Scribble)
- Convert the model to OpenVINO Intermediate Representation (IR)
    - QFormer
    - Text encoder
    - ControlNet
    - UNet
    - Variational Autoencoder (VAE)
    - Select inference device
- Inference
    - Zero-Shot subject-driven generation
    - Controlled subject-driven generation (Canny-edge)
    - Controlled subject-driven generation (Scribble)
- Interactive inference


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
