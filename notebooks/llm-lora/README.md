# Text Generation with LoRA via OpenVINO GenAI

LoRA, or [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685), is a popular and lightweight training technique used for fine-tuning Large Language and Stable Diffusion Models without needing full model training. Full fine-tuning of larger models (consisting of billions of parameters) is inherently expensive and time-consuming. LoRA works by adding a smaller number of new weights to the model for training, rather than retraining the entire parameter space of the model. This makes training with LoRA much faster, memory-efficient, and produces smaller model weights (a few hundred MBs), which are easier to store and share.

At its core, LoRA leverages the concept of low-rank matrix factorization. Instead of updating all the parameters in a neural network, LoRA decomposes the parameter space into two low-rank matrices. This decomposition allows the model to capture essential information with fewer parameters, significantly reducing the amount of data and computation required for fine-tuning. This vastly reduces the storage requirement for large language models adapted to specific tasks and enables efficient task-switching during deployment all without introducing inference latency. 

![](https://github.com/user-attachments/assets/bf823c71-13b4-402c-a7b4-d6fc30a60d88)

Some more advantages of using LoRA:

* LoRA makes fine-tuning more efficient by drastically reducing the number of trainable parameters.
* The original pre-trained weights are kept frozen, which means you can have multiple lightweight and portable LoRA models for various downstream tasks built on top of them.
* LoRA is orthogonal to many other parameter-efficient methods and can be combined with many of them.
* Performance of models fine-tuned using LoRA is comparable to the performance of fully fine-tuned models.
* LoRA does not add any inference latency because adapter weights can be merged with the base model.

More details about LoRA can be found in HuggingFace [conceptual guide](https://huggingface.co/docs/peft/conceptual_guides/lora) and [blog post](https://huggingface.co/blog/peft).
  
In this tutorial we explore possibilities to use LoRA with OpenVINO Generative API.

## Notebook Contents

This notebook demonstrates how to perform generate  using OpenVINO GenAI and LoRA adapters.

The tutorial consists of following steps:
- Load and configure LoRA adapters
- Run inference with OpenVINO GenAI LLMPipeline

## Installation Instructions
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llm-lora/README.md" />
