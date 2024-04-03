# Visual-language assistant with LLaVA Next and OpenVINO

[LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) is new generation of LLaVA model family that marks breakthrough in advanced language reasoning over images, introducing improved OCR and expanded world knowledge. [LLaVA](https://llava-vl.github.io) (Large Language and Vision Assistant) is large multimodal model that aims to develop a general-purpose visual assistant that can follow both language and image instructions to complete various real-world tasks. The idea is to combine the power of large language models (LLMs) with vision encoders like CLIP to create an end-to-end trained neural assistant that understands and acts upon multimodal instructions.

In this tutorial we consider how to convert and optimize LLaVa-NeXT model from Transformers library for creating multimodal chatbot. We will utilize the power of [llava-v1.6-mistral-7b](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) model for creating multimodal chatbot, but the simmilar actions are also applicable to other models of LLaVA family compatible with HuggingFace transformers implementation. Additionally, we demonstrate how to apply stateful transformation on LLM part and model optimization techniques like weights compression and quantization using [NNCF](https://github.com/openvinotoolkit/nncf) 

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Download PyTorch model
- Convert model to OpenVINO Intermediate Representation (IR)
- Compress Language Model weights
- Quantize Image Encoder
- Prepare Inference Pipeline
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content. Image bellow shows a result of model work.
![](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/51263b7c-fff4-454b-b347-fb16bdc7673)


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).