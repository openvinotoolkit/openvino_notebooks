# Text-to-image generation with Z-Image-Turbo and OpenVINO

Z-Image-Turbo is Alibaba's production-ready, open-source 6B-parameter image generation model from the Z-Image family.

<img width="2000" height="931" alt="image" src="https://github.com/user-attachments/assets/24e4ab2a-febe-4496-bd53-17e139cfc410" />

**Highlights**

- Photorealistic quality + strong bilingual (Chinese & English) text rendering
- Excellent instruction-following and in-context editing (supports bounding boxes, object-level control)
- Uses Single-Stream Diffusion Transformer (S3-DiT): text and image tokens processed in one unified stream
- Prompt Enhancer (PE) + Decoupled DMD/DMDR distillation for high-quality 1–8 step generation

More details about model can be found in [paper post](https://arxiv.org/pdf/2511.22699) and [model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).

In this tutorial we consider how to convert and optimize Z-Image-Turbo model using OpenVINO.

### Notebook Contents

In this demonstration, you will learn how to perform text-to-image generation using Z-Image-Turbo and OpenVINO. 

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

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO and is using a custom branch of optimum-intel. It may be fully supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/z-image-turbo/README.md" />
