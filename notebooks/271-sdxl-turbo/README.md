# Single-step image generation using SDXL-turbo and OpenVINO

SDXL-Turbo is a fast generative text-to-image model that can synthesize photorealistic images from a text prompt in a single network evaluation. SDXL-Turbo is a distilled version of [SDXL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), trained for real-time synthesis. 
SDXL Turbo is based on a novel distillation technique called Adversarial Diffusion Distillation (ADD), which enables the model to synthesize image outputs in a single step and generate real-time text-to-image outputs while maintaining high sampling fidelity. More details about this distillation approach can be found in [technical report](https://stability.ai/research/adversarial-diffusion-distillation). More details about model can be found in [Stability AI blog post](https://stability.ai/news/stability-ai-sdxl-turbo).

Previously, we already discussed how to launch Stable Diffusion XL model using OpenVINO in the following [notebook](../248-stable-diffusion-xl), in this tutorial we will focus on the [SDXL-turbo](https://huggingface.co/stabilityai/sdxl-turbo) version. Additionally, to improve image decoding speed, we will use [Tiny Autoencoder](https://github.com/madebyollin/taesd), which is useful for real-time previewing of the SDXL generation process.

We will use a pre-trained model from the [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index) library. To simplify the user experience, the [Hugging Face Optimum Intel](https://huggingface.co/docs/optimum/intel/index) library is used to convert the models to OpenVINO™ IR format.

The notebook provides a simple interface that allows communication with a model using text instruction. In this demonstration user can provide input instructions and the model generates an image. An additional part demonstrates how to run quantization with [NNCF](https://github.com/openvinotoolkit/nncf/) to speed up pipeline.

The image below illustrates the provided generated image example.

![text2img_example.png](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/79b625c7-0f0a-4f19-8e38-e6f896f75c3e)

>**Note**: Some demonstrated models can require at least 32GB RAM for conversion and running.

### Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Convert model to OpenVINO intermediate representation (IR) format
- Run Text-to-Image generation
- Run Image-to-Image generation
- Optimize model with [NNCF](https://github.com/openvinotoolkit/nncf/) quantization
- Compare results of original and optimized pipelines
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
