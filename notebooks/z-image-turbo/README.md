# Text-to-image generation with Z-Image-Turbo and OpenVINO

Z-Image-Turbo is Alibaba’s production-ready, open-source 6B-parameter image generation model from the Z-Image family.

<p align="center">
    <img src="https://github.com/Tongyi-MAI/Z-Image/blob/main/assets/showcase.jpg" width="90%"/>
<p>

**Highlights**

- Sub-second inference (< 1 s) on a single consumer GPU (≤ 16 GB VRAM, e.g. RTX 4090)
- Photorealistic quality + strong bilingual (Chinese & English) text rendering
- Excellent instruction-following and in-context editing (supports bounding boxes, object-level control)
- Uses Single-Stream Diffusion Transformer (S3-DiT): text and image tokens processed in one unified stream
- Prompt Enhancer (PE) + Decoupled DMD/DMDR distillation for high-quality 1–8 step generation

More details about model can be found in [blog post](https://bfl.ai/announcements/flux-1-kontext-dev) and [model card](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).

In this tutorial we consider how to convert and optimize Z-Image-Turbo model using OpenVINO.

>**Note**: Some demonstrated models can require at least 32GB RAM for conversion and running.

### Notebook Contents

In this demonstration, you will learn how to perform image-to-image generation using Flux.1 Kontext and OpenVINO. 

The tutorial consists of the following steps:

- Install prerequisites
- Collect Pytorch model pipeline
- Convert model to OpenVINO intermediate representation (IR) format 
- Compress weights using NNCF
- Prepare OpenVINO Inference pipeline
- Run Text-to-Image generation
- Launch interactive demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/flux.1-kontext/README.md" />
