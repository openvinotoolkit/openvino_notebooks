# Visual-language assistant with LLaVA Next and OpenVINO

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/nano-llava-multimodal-chatbot/nano-llava-multimodal-chatbot.ipynb)

nanoLLaVA is a "small but mighty" 1B vision-language model designed to run efficiently on edge devices. It uses [SigLIP-400m](https://huggingface.co/google/siglip-so400m-patch14-384) as Image Encoder and [Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B) as LLM.
In this tutorial, we consider how to convert and run nanoLLaVA model using OpenVINO. Additionally, we will optimize model  using [NNCF](https://github.com/openvinotoolkit/nncf)

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Download PyTorch model
- Convert model to OpenVINO Intermediate Representation (IR)
- Compress model weights using NNCF
- Prepare Inference Pipeline
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content.


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/nano-llava-multimodal-chatbot/README.md" />
