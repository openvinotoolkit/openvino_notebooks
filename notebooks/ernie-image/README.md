# Text-to-Image Generation with ERNIE-Image-Turbo and OpenVINO

[ERNIE-Image-Turbo](https://huggingface.co/Baidu/ERNIE-Image-Turbo) is Baidu's production-ready, open-source image generation model based on the ERNIE family.

<img width="1878" height="914" alt="image" src="https://github.com/user-attachments/assets/d691f512-4a5d-4bca-9620-835cd2d5502a" />

**Highlights**

- High-quality photorealistic image generation with strong bilingual (Chinese & English) support
- Uses Diffusion Transformer architecture with Mistral3 text encoder
- Optional **Prompt Enhancer (PE)** — a built-in language model that automatically expands short prompts into detailed visual descriptions
- Fast 8-step generation with flow matching scheduler

More details about the model can be found in the [model card](https://huggingface.co/Baidu/ERNIE-Image-Turbo).

In this tutorial we consider how to convert and optimize ERNIE-Image-Turbo model using OpenVINO.

### Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Convert model to OpenVINO IR format with optional INT4 weight compression
- Optionally export the Prompt Enhancer (PE) model
- Run text-to-image generation with OpenVINO
- Launch interactive Gradio demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO and is using a custom branch of optimum-intel. It may be fully supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/ernie-image/README.md" />
