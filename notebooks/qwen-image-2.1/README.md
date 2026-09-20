# Qwen-Image 2.1 image generation with OpenVINO

Qwen-Image 2.1 is a unified image generation model that supports text-to-image generation and image-conditioned editing in one pipeline. The prompt and condition images are encoded together by Qwen3-VL and processed by a block-causal diffusion transformer.

For model architecture and usage details, see the [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) model card and the [Qwen-Image source repository](https://github.com/QwenLM/Qwen-Image).

This tutorial demonstrates how to:

- download and convert [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) to OpenVINO IR;
- run text-to-image generation with OpenVINO GenAI;
- run image-conditioned editing with the same exported model;
- save generated images with reproducible configuration details in their filenames;
- measure pipeline loading, first-run, and warm-run latency;
- launch an interactive demo with pipeline, precision, and device selection.

> **Important:** Qwen-Image 2.1 image conditioning is not classic image-to-image generation based on adding noise to an initial image. The condition image is part of the multimodal context, so this notebook intentionally does not expose a `strength` parameter.

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO. It may be fully supported and validated in the future.

## Notebook Contents

1. Install the latest stable Gradio, PyTorch, NNCF, and utility packages; upstream Diffusers and Optimum Intel branches; and OpenVINO nightly builds, with a temporary cross-platform fallback that automatically extracts cached or manually downloaded OpenVINO GenAI main-branch artifact ZIPs
2. Select export options
3. Convert the model to OpenVINO IR
4. Run text-to-image generation
5. Run image-conditioned editing
6. Benchmark both scenarios
7. Launch an interactive demo with dynamic pipeline, FP16/INT8/INT4, and device selection

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to the [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen-image-2.1/README.md" />
