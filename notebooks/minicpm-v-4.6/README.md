# Visual-language assistant with MiniCPM-V 4.6 and OpenVINO

[MiniCPM-V 4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) is the most edge-deployment-friendly model in the MiniCPM-V series. Built on SigLIP2-400M and Qwen3.5-0.8B LLM, it inherits the strong single-image, multi-image, and video understanding capabilities of the MiniCPM-V family while significantly improving computation efficiency.

**Highlights**

- **Leading Foundation Capability** — Scores 13 on Artificial Analysis Intelligence Index, outperforming Qwen3.5-0.8B (10) with 19× fewer tokens
- **Strong Multimodal Capability** — Outperforms Qwen3.5-0.8B on most vision-language tasks, reaching Qwen3.5 2B-level on OpenCompass, RefCOCO, HallusionBench, and OCRBench
- **Ultra-Efficient Architecture** — Based on LLaVA-UHD v4, reduces visual encoding FLOPs by 50%+; supports mixed 4×/16× visual token compression
- **Broad Mobile Platform Coverage** — Deployable on iOS, Android, and HarmonyOS

More details about the model can be found in the [model card](https://huggingface.co/openbmb/MiniCPM-V-4.6) and the original [repo](https://github.com/OpenBMB/MiniCPM-V).

In this tutorial we consider how to convert and optimize MiniCPM-V 4.6 model for creating a multimodal chatbot using [Optimum Intel](https://github.com/huggingface/optimum-intel) and [OpenVINO](https://github.com/openvinotoolkit/openvino).

### Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Select model and weight format
- Convert and optimize model using Optimum Intel CLI
- Run model inference with OpenVINO
- Launch interactive Gradio demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO and is using a custom branch of optimum-intel. It may be fully supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/minicpm-v-4.6/README.md" />
