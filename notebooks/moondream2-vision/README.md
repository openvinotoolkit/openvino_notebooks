# Visual-Language Assistant with Moondream2 and OpenVINO

[Moondream2](https://huggingface.co/vikhyatk/moondream2) is a small (2B parameters) vision-language model designed to run efficiently on edge devices. Despite its compact size, it supports a wide range of vision-language tasks including image captioning, visual question answering, object detection, pointing, and OCR.

This notebook demonstrates how to convert Moondream2 to OpenVINO IR format with INT4 weight compression and run inference for multiple tasks.

The tutorial consists of the following steps:
- Install prerequisites
- Convert and optimize the model using the Optimum CLI with INT4 weight compression
- Run inference for image captioning, visual question answering, and object detection
- Launch an interactive Gradio demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/moondream2-vision/README.md" />
