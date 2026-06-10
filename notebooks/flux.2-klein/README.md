# FLUX.2 Klein Image Generation with OpenVINO™

[FLUX.2 [klein]](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) is a 4-billion-parameter rectified flow transformer from Black Forest Labs. It unifies text-to-image generation and image editing in a single compact architecture, delivering state-of-the-art quality with end-to-end inference in as low as 4 steps.

Key features:
- Sub-second generation with distilled 4-step mode
- Text-to-image and multi-reference image editing in one model
- Runs on consumer GPUs (~13 GB VRAM)
- Open weights under Apache 2.0 license

In this notebook we demonstrate how to convert and optimize FLUX.2 [klein] 4B using OpenVINO with INT4 weight compression.

## Notebook Contents

The notebook covers:
1. Prerequisites and installation
2. Model export using optimum-cli with INT4 weight compression
3. Inference with OpenVINO on CPU or GPU
4. Text-to-image generation
5. Image editing with reference images
6. Interactive Gradio demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/flux.2-klein/README.md" />
