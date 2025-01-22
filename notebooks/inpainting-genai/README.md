# Inpainting with OpenVINO GenAI

inpainting is task of replacing or editing specific area of input image. This makes it a useful tool for image restoration like removing defects and artifacts, or even replacing an image area with something entirely new. Inpainting relies on a mask to determine which regions of an image to fill in; the area to inpaint is represented by white pixels and the area to keep is represented by black pixels. The white pixels are filled in by the prompt.

In this tutorial we consider how to perform inpainting using OpenVINO GenAI.

## About OpenVINO GenAI

[OpenVINO™ GenAI](https://github.com/openvinotoolkit/openvino.genai) is a library of the most popular Generative AI model pipelines, optimized execution methods, and samples that run on top of highly performant OpenVINO Runtime.

This library is friendly to PC and laptop execution, and optimized for resource consumption. It requires no external dependencies to run generative models as it already includes all the core functionality (e.g. tokenization via openvino-tokenizers).

OpenVINO GenAI supports popular diffusion models like Stable Diffusion or SDXL for performing image generation. You can find supported models list in [OpenVINO GenAI documentation](https://github.com/openvinotoolkit/openvino.genai/blob/master/SUPPORTED_MODELS.md#image-generation-models). Previously, we considered how to run text-to-image and image-to-image generation with OpenVINO GenAI and apply multiple LoRA adapters, now is inpainting turn.

## About OpenVINO GenAI

[OpenVINO™ GenAI](https://github.com/openvinotoolkit/openvino.genai) is a library of the most popular Generative AI model pipelines, optimized execution methods, and samples that run on top of highly performant OpenVINO Runtime.

This library is friendly to PC and laptop execution, and optimized for resource consumption. It requires no external dependencies to run generative models as it already includes all the core functionality (e.g. tokenization via openvino-tokenizers).

OpenVINO GenAI supports popular diffusion models like Stable Diffusion or SDXL for performing image generation. You can find supported models list in [OpenVINO GenAI documentation](https://github.com/openvinotoolkit/openvino.genai/blob/master/SUPPORTED_MODELS.md#image-generation-models). Previously, we considered how to run text-to-image generation with OpenVINO GenAI and apply multiple LoRA adapters, mow is image-to-image. 

## Notebook Contents

In this notebook we will demonstrate how to use Latent Diffusion models like Stable Diffusion 1.5, 2.1, SDXL for inpainting using OpenVINO GenAI InpaintingPipeline. 
All it takes is two steps: 
1. Export OpenVINO IR format model using the [Hugging Face Optimum](https://huggingface.co/docs/optimum/installation) library accelerated by OpenVINO integration.
The Hugging Face Optimum Intel API is a high-level API that enables us to convert and quantize models from the Hugging Face Transformers library to the OpenVINO™ IR format. For more details, refer to the [Hugging Face Optimum Intel documentation](https://huggingface.co/docs/optimum/intel/inference).
1. Run inference using the standard [Inpainting pipeline](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html) from OpenVINO GenAI.

The tutorial consists of following steps:
- Install prerequisites
- Prepare models
  - Convert models using HuggingFace Optimum Intel
  - Obtain optimized models from HuggingFace Hub
- Prepare Inference pipeline
- Run inpainting
- Explore advanced options for generation results improvement
- Launch interactive demo

![inpaint_diagram](https://github.com/user-attachments/assets/f74f7b65-2892-49b5-b13f-4038dad2bc90)


## Installation Instructions
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/inpainting-genai/README.md" />
