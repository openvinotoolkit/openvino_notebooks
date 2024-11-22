# Visual-language assistant with LLaVA Next and OpenVINO

[LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) is new generation of LLaVA model family that marks breakthrough in advanced language reasoning over images, introducing improved OCR and expanded world knowledge. [LLaVA](https://llava-vl.github.io) (Large Language and Vision Assistant) is large multimodal model that aims to develop a general-purpose visual assistant that can follow both language and image instructions to complete various real-world tasks. The idea is to combine the power of large language models (LLMs) with vision encoders like CLIP to create an end-to-end trained neural assistant that understands and acts upon multimodal instructions.

In this tutorial we consider how to convert and optimize LLaVA-NeXT model from Transformers library for creating multimodal chatbot. We will utilize the power of [llava-v1.6-mistral-7b](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) model for creating multimodal chatbot, but the similar actions are also applicable to other models of LLaVA family compatible with HuggingFace transformers implementation. Additionally, we demonstrate how to apply stateful transformation on LLM part and model optimization techniques like weights compression and quantization using [NNCF](https://github.com/openvinotoolkit/nncf) 

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert model to OpenVINO Intermediate Representation (IR) using Optimum CLI
- Compress Language Model weights using NNCF
- Prepare OpenVINO GenAI Inference Pipeline
- Run OpenVINO GenAI model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content. Image bellow shows a result of model work.
![](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/a562e9de-5b94-4e24-ac52-532019fc92d3)


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llava-next-multimodal-chatbot/README.md" />
