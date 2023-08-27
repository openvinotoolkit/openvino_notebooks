# Image Generation with Tiny-SD
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/251-tiny-sd-image-generation/251-tiny-sd-image-generation.ipynb)

In recent times, the AI community has witnessed a remarkable surge in the development of larger and more performant language models, such as Falcon 40B, LLaMa-2 70B, Falcon 40B, MPT 30B, and in the imaging domain with models like SD2.1 and SDXL. These advancements have undoubtedly pushed the boundaries of what AI can achieve, enabling highly versatile and state-of-the-art image generation and language understanding capabilities. However, the breakthrough of large models comes with substantial computational demands. To resolve this issue, recent research on efficient Stable Diffusion has prioritized reducing the number of sampling steps and utilizing network quantization.

Moving towards the goal of making image generative models faster, smaller, and cheaper, Tiny-SD was proposed by Segmind. Tiny SD is a compressed Stable Diffusion (SD) model that has been trained on Knowledge-Distillation (KD) techniques and the work has been largely based on this [paper](https://arxiv.org/pdf/2305.15798.pdf). The authors describe a Block-removal Knowledge-Distillation method where some of the UNet layers are removed and the student model weights are trained. Using the KD methods described in the paper, they were able to train two compressed models using the 🧨 diffusers library; Small and Tiny, that have 35% and 55% fewer parameters, respectively than the base model while achieving comparable image fidelity as the base model. More details about model can be found in [model card](https://huggingface.co/segmind/tiny-sd), [blog post](https://huggingface.co/blog/sd_distillation) and training [repository](https://github.com/segmind/distill-sd).

This notebook demonstrates how to convert and run the Tiny-SD model using OpenVINO.
It considers two approaches of image generation using an AI method called `diffusion`:

* `Text-to-image` generation to create images from a text description as input.
* `Text-guided Image-to-Image` generation to create an image, using text description and initial image semantic.

The complete pipeline of this demo is shown below.

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/260981188-c112dd0a-5752-4515-adca-8b09bea5d14a.png"/>
</p>


This is a demonstration in which you can type a text description (and provide input image in case of Image-to-Image generation) and the pipeline will generate an image that reflects the context of the input text.
Step-by-step, the diffusion process will iteratively denoise latent image representation while being conditioned on the text embeddings provided by the text encoder.

The following image shows an example of the input sequence and corresponding predicted image.

**Input text:** RAW studio photo of An intricate forest minitown landscape trapped in a bottle, atmospheric oliva lighting, on the table, intricate details, dark shot, soothing tones, muted colors

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/260904650-274fc2f9-24d2-46a3-ac3d-d660ec3c9a19.png"/>
</p>

## Notebook Contents

This notebook demonstrates how to convert and run [Tiny-SD](https://huggingface.co/segmind/tiny-sd) using OpenVINO.

The notebook contains the following steps:

1. Convert PyTorch models to OpenVINO Intermediate Representation using OpenVINO Converter Tool (OVC).
2. Prepare Inference Pipeline.
3. Run Inference pipeline with OpenVINO.
4. Run Interactive demo for Tiny-SD model

The notebook also provides interactive interface for image generation based on user input (text prompts and source image, if required).
**Text-to-Image Generation Example**
![text2img.png](https://user-images.githubusercontent.com/29454499/260905732-f291d316-8835-4872-8d9b-8a1214448bfd.png)

**Image-to-Image Generation Example**
![img2img.png](https://user-images.githubusercontent.com/29454499/260905907-4b7835c6-1f63-4d00-a1ec-ccc4d7fca182.png)



## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).