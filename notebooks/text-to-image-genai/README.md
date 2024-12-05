# Text to Image pipeline and OpenVINO with Generate API

[OpenVINO™ GenAI](https://github.com/openvinotoolkit/openvino.genai) is a library of the most popular Generative AI model pipelines, optimized execution methods, and samples that run on top of highly performant OpenVINO Runtime.

This library is friendly to PC and laptop execution, and optimized for resource consumption. It requires no external dependencies to run generative models as it already includes all the core functionality (e.g. tokenization via openvino-tokenizers).

In this tutorial we consider how to use OpenVINO GenAI for image generation scenario.

## Notebook Contents

In this notebook we will demonstrate how to use text to image models like Stable Diffusion 1.5, 2.1, LCM using [Dreamlike Anime 1.0](https://huggingface.co/dreamlike-art/dreamlike-anime-1.0) as an example. All it takes is two steps: 
1. Export OpenVINO IR format model using the [Hugging Face Optimum](https://huggingface.co/docs/optimum/installation) library accelerated by OpenVINO integration.
The Hugging Face Optimum Intel API is a high-level API that enables us to convert and quantize models from the Hugging Face Transformers library to the OpenVINO™ IR format. For more details, refer to the [Hugging Face Optimum Intel documentation](https://huggingface.co/docs/optimum/intel/inference).
2. Run inference using the standard [Text-to-Image Generation pipeline](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html) from OpenVINO GenAI.

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