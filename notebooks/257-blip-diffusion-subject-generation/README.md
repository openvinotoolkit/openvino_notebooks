# Subject-driven image generation and editing using BLIP Diffusion and OpenVINO
![](https://github.com/salesforce/LAVIS/raw/main/projects/blip-diffusion/teaser-website.png)
Blip Diffusion is a text-to-image diffusion model introduced in the paper [\"BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing\" by Li et al. (2023)](https://arxiv.org/abs/2305.14720). It is a unique diffusion model, pre-trained on a multimodal dataset of text and images, enabling it to generate images that align more coherently with the subject of the text prompt.

Blip Diffusion undergoes a two-stage training process. In the initial stage, a multimodal encoder is pre-trained to acquire a representation of the text and images within the dataset. This encoder is specifically trained to capture informative features encompassing both the visual content of the images and the semantic content of the text.

In the second stage, the pre-trained multimodal encoder is employed to train a diffusion model. This diffusion model learns to generate images that closely correspond to the multimodal representation derived from the text prompt.

Once the diffusion model completes its training, it becomes capable of generating images from text prompts. The image generation process commences with a random noise image, progressively integrating more details until it faithfully mirrors the multimodal representation inherent in the text prompt.

Blip Diffusion offers several advantages over other text-to-image diffusion models. First, it excels at generating images that harmonize with the subject matter of the text prompt, owing to its pre-training on a multimodal dataset of text and images, which facilitates the understanding of the relationship between textual and visual elements.

Second, Blip Diffusion demonstrates a remarkable ability to produce diverse images. Unlike models reliant on fixed image templates, Blip Diffusion crafts images from scratch by leveraging the multimodal representation derived from the text prompt, fostering uniqueness in its outputs.

Third, Blip Diffusion excels at generating realistic images due to its training on a dataset containing real-world images.

In summary, Blip Diffusion stands as a potent text-to-image diffusion model for generating high-quality images from text prompts. It distinguishes itself through its capacity to produce images that align closely with the text prompt's subject, its versatility in generating diverse visuals from scratch, and its ability to craft realistic images. In this tutorial we will show how to convert Blip Diffusion model to OpenVINO Intermediate Representation and perform inference.

## Notebook contents
The tutorial consists of the following steps:

- Prerequisites
- Load the model
- Infer original model
- Convert model to OpenVINO Intermediate Representation (IR)
- Inference
- Interactive inference


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).