# Document parsing with GLM-OCR and OpenVINO

[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) is a compact 0.9B-parameter
OCR vision-language model released by Zhipu AI. It combines a CogViT-derived
vision encoder with a GLM-4.5-style text decoder and is specialised for three
document-parsing tasks: **Text Recognition**, **Formula Recognition**, and
**Table Recognition**. On OmniDocBench v1.5 it reaches 94.62 points, making
it competitive with much larger commercial OCR systems.

<img width="1071" height="405" alt="image" src="https://github.com/user-attachments/assets/3394dd11-cf7f-43d9-a77e-920169d0f099" />

You can find more information in the
[model card](https://huggingface.co/zai-org/GLM-OCR).

In this tutorial we convert GLM-OCR to the OpenVINO Intermediate
Representation, compress it to 4-bit weights with [NNCF](https://github.com/openvinotoolkit/nncf),
and run inference on Intel CPU / integrated GPU / Arc GPU through
[optimum-intel](https://huggingface.co/docs/optimum/intel/index). A Gradio
demo mirroring the official [GLM-OCR-Demo](https://huggingface.co/spaces/prithivMLmods/GLM-OCR-Demo)
Space is provided, plus an optional PP-DocLayout-V3 pipeline that detects
document regions and dispatches them to GLM-OCR with the matching prompt.

## Notebook contents
The tutorial consists of the following steps:

- Install requirements
- Convert and Optimize model (FP16 or INT4 weights)
- Select inference device (CPU / GPU / AUTO)
- Run GLM-OCR on a sample image (Text / Formula / Table prompts)
- Optional: PP-DocLayout-V3 + GLM-OCR document-parsing pipeline
- Launch Interactive Gradio demo

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a
Jupyter server to start. For details, please refer to the
[Installation Guide](../../README.md).

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with
OpenVINO and is using a custom branch of optimum-intel. It may be fully
supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/glm-ocr/README.md" />
