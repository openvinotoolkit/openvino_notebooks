# Text to Image pipeline and OpenVINO with Generate API

OpenVINO GenAI is a new flavor of OpenVINO, aiming to simplify running inference of generative AI models. It hides the complexity of the generation process and minimizes the amount of code required. You can now provide a model and input context directly to OpenVINO, which performs tokenization of the input text, executes the generation loop on the selected device, and returns the generated results. For a quick start guide, refer to the [GenAI API Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html).

In this notebook we will demonstrate how to use text to image models like Stable Diffusion 1.5, 2.1, LCM using [Dreamlike Anime 1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0) as an example. All it takes is two steps: 
1. Export OpenVINO IR format model using the [Hugging Face Optimum](https://huggingface.co/docs/optimum/installation) library accelerated by OpenVINO integration.
The Hugging Face Optimum Intel API is a high-level API that enables us to convert and quantize models from the Hugging Face Transformers library to the OpenVINO™ IR format. For more details, refer to the [Hugging Face Optimum Intel documentation](https://huggingface.co/docs/optimum/intel/inference).
2. Run inference using the standard [Text to Image pipeline](https://openvino-doc.iotg.sclab.intel.com/nightly/learn-openvino/llm_inference_guide/genai-guide/genai-use-cases.html#using-genai-for-text-to-image-generation) from OpenVINO GenAI.

## Notebook Contents

This notebook demonstrates how to perform automatic speech recognition (ASR) using the Whisper model and OpenVINO.

The tutorial consists of following steps:
- Prerequisites
- Convert model using Optimum-CLI tool
- Run inference OpenVINO model with Text2ImagePipeline
- Run inference OpenVINO model with Text2ImagePipeline with optional LoRA adapters
- Interactive demo


## Installation Instructions
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/text-to-image-genai/README.md" />