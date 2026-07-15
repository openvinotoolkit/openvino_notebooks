# Document parsing with HunyuanOCR and OpenVINO

[HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) is a lightweight,
end-to-end OCR-specialised vision-language model released by the Tencent Hunyuan
team. It pairs a SigLIP-style vision encoder with a compact HunYuan text decoder
and unifies **document parsing, text spotting, information extraction, and
text-image translation** in a single end-to-end VLM, while remaining small
enough for on-device deployment.

You can find more information in the
[model card](https://huggingface.co/tencent/HunyuanOCR) and the
[GitHub repository](https://github.com/Tencent-Hunyuan/HunyuanOCR).

In this tutorial we convert HunyuanOCR to the OpenVINO Intermediate
Representation via [optimum-intel](https://huggingface.co/docs/optimum/intel/index),
optionally compress its weights to **INT8** (or keep **FP16**) with
[NNCF](https://github.com/openvinotoolkit/nncf), and run inference on Intel
CPU / integrated GPU / Arc GPU. The basic inference usage is aligned with the
official HuggingFace `transformers` (native) snippet from the model card. A
streaming Gradio demo — with the same task presets as the official
[HunyuanOCR Space](https://huggingface.co/spaces/tencent/HunyuanOCR) — is
provided as well.

## Notebook contents
The tutorial consists of the following steps:

- Install requirements
- Convert and Optimize model (INT8 or FP16 weights)
- Select inference device (CPU / GPU / AUTO)
- Run HunyuanOCR on a sample image (document parsing / spotting / IE / translation prompts)
- Launch an interactive Gradio demo

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a
Jupyter server to start. For details, please refer to the
[Installation Guide](../../README.md).

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with
OpenVINO and is using a custom branch of optimum-intel
(`hunyuan-ocr-support`). It may be fully supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/hunyuan-ocr/README.md" />
