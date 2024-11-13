# Multi LoRA Image Generation

LoRA, or [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685), is a popular and lightweight training technique used for fine-tuning Large Language and Stable Diffusion Models without needing full model training. Full fine-tuning of larger models (consisting of billions of parameters) is inherently expensive and time-consuming. LoRA works by adding a smaller number of new weights to the model for training, rather than retraining the entire parameter space of the model. This makes training with LoRA much faster, memory-efficient, and produces smaller model weights (a few hundred MBs), which are easier to store and share.

At its core, LoRA leverages the concept of low-rank matrix factorization. Instead of updating all the parameters in a neural network, LoRA decomposes the parameter space into two low-rank matrices. This decomposition allows the model to capture essential information with fewer parameters, significantly reducing the amount of data and computation required for fine-tuning.

![](https://github.com/user-attachments/assets/bf823c71-13b4-402c-a7b4-d6fc30a60d88)

By incorporating LoRA into Stable Diffusion models, we can enhance their ability to understand complex relationships and patterns in data.  This approach opens up numerous possibilities:
* **Art and Design**: Artists can fine-tune models to generate images that align with their unique styles, creating personalized artwork effortlessly.
* **Content Creation**: Businesses can customize image generation models to produce branded visuals, enhancing marketing and media production.
* **Entertainment**: Game developers and filmmakers can use fine-tuned models to create realistic and imaginative worlds, streamlining the creative process.
  
In this tutorial we explore possibilities to use LoRA with OpenVINO Generative API.

## Notebook Contents

This notebook demonstrates how to perform image generation using OpenVINO GenAI and LoRA adapters.

The tutorial consists of following steps:
- Convert model using Optimum Intel
- Load and configure LoRA adapters
- Run inference with OpenVINO GenAI Text2ImagePipeline
- Interactive demo

In interactive demonstration you can try to generate images using Stable Diffusion XL and make stylization using different LoRA adapters. Example of generated images with the same prompt and different adapters can be found bellow.

![](https://github.com/user-attachments/assets/2138a109-add0-473a-b9dc-015e4b0415ce)


## Installation Instructions
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/multilora-image-generation/README.md" />
