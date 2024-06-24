# Image generation with Stable Diffusion v3 and OpenVINO

Stable Diffusion V3 is next generation of latent diffusion image Stable Diffusion models family that  outperforms state-of-the-art text-to-image generation systems in typography and prompt adherence, based on human preference evaluations. In comparison with previous versions, it based on Multimodal Diffusion Transformer (MMDiT) text-to-image model that features greatly improved performance in image quality, typography, complex prompt understanding, and resource-efficiency.

![mmdit.png](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/dd079427-89f2-4d28-a10e-c80792d750bf)

More details about model can be found in [model card](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [research paper](https://stability.ai/news/stable-diffusion-3-research-paper) and [Stability.AI blog post](https://stability.ai/news/stable-diffusion-3-medium).
In this tutorial, we will consider how to convert and optimize Stable Diffusion v3 for running with OpenVINO.
If you want to run previous Stable Diffusion versions, please check our other notebooks:

* [Stable Diffusion](../stable-diffusion-text-to-image)
* [Stable Diffusion v2](../stable-diffusion-v2)
* [Stable Diffusion XL](../stable-diffusion-xl)
* [LCM Stable Diffusion](../latent-consistency-models-image-generation)
* [Turbo SDXL](../sdxl-turbo)
* [Turbo SD](../sketch-to-image-pix2pix-turbo)


The notebook provides a simple interface that allows communication with a model using text instruction. In this demonstration user can provide input instructions and the model generates an image. An additional part demonstrates how to optimize model with [NNCF](https://github.com/openvinotoolkit/nncf/) to speed up pipeline and reduce memory consumption.

The image below illustrates the provided generated image example.

![text2img_example.png](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/ac99098c-66ec-4b7b-9e01-e80625f1dc3f)

>**Note**: Some demonstrated models can require at least 32GB RAM for conversion and running.

### Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Collect Pytorch model pipeline
- Convert model to OpenVINO intermediate representation (IR) format and compress weights using NNCF
- Prepare OpenVINO Inference pipeline
- Run Text-to-Image generation
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
