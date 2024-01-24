# Image Generation with Stable Diffusion and IP-Adapter

[IP-Adapter](https://hf.co/papers/2308.06721) is an effective and lightweight adapter that adds image prompting capabilities to a diffusion model. This adapter works by decoupling the cross-attention layers of the image and text features. All the other model components are frozen and only the embedded image features in the UNet are trained. As a result, IP-Adapter files are typically only ~100MBs.
![ip-adapter-pipe.png](https://huggingface.co/h94/IP-Adapter/resolve/main/fig1.png)

In this tutorial, we will consider how to convert and run Stable Diffusion pipeline with loading IP-Adapter. We will use [stable-diffusion-v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) as base model and apply official [IP-Adapter](https://huggingface.co/h94/IP-Adapter) weights. Also for speedup generation process we will use [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)

We will use a pre-trained model from the [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index) library.

The notebook provides a simple interface that allows communication with a model using text instruction and images. In this demonstration user can provide input instructions and image for ip-adapter and the model generates an image. 
The image below illustrates the provided generated image example.

![text2img_example.gif](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/e07f21a6-a7cf-49cf-bd2f-ec26cb7276d6)

>**Note**: Some demonstrated models can require at least 32GB RAM for conversion and running.

### Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Prepare Diffusers pipeline
- Convert PyTorch models
- Prepare OpenVINO inference pipeline
- Run model inference
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
