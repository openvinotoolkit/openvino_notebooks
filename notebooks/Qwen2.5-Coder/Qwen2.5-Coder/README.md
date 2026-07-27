# Qwen2.5-Coder with OpenVINO™

<!-- TODO: replace with an actual screenshot of the Gradio coding-assistant demo before opening the PR, e.g.:
![qwen2.5-coder-demo](https://github.com/user-attachments/assets/<upload-id>)
-->

Qwen2.5-Coder is a family of code-specialized large language models from Alibaba's Qwen team, built on top of Qwen2.5 and further pretrained on 5.5 trillion tokens of code, text and synthetic data. Key features:

* **Strong Code Generation**: State-of-the-art coding capability among open-source models of comparable size, matching the coding performance of much larger LLMs.
* **Multi-language Support**: Understands and generates code across 90+ programming languages, including Python, JavaScript, TypeScript, Java, and C++.
* **Long Context**: Supports up to a 32K token context window, enabling work with large codebases and complex, multi-file functions.
* **Instruction Following**: Fine-tuned for code completion, code generation, code reasoning and repair, and general coding assistant scenarios.

More details can be found in the [Qwen2.5-Coder blog](https://qwenlm.github.io/blog/qwen2.5-coder-family/) and [model card](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct).

In this tutorial we consider how to convert and optimize Qwen2.5-Coder model for creating a coding assistant using [Optimum Intel](https://github.com/huggingface/optimum-intel). Additionally, we demonstrate how to apply weight compression to the model using [NNCF](https://github.com/openvinotoolkit/nncf).

This folder contains two notebooks:

### [qwen2.5-coder.ipynb](qwen2.5-coder.ipynb)

The main tutorial. In this notebook, we will:
1. Export Qwen2.5-Coder to OpenVINO IR format with INT4 weight compression using Optimum Intel.
2. Run inference with `OVModelForCausalLM` using the drop-in Hugging Face `generate()` API.
3. Build an interactive Gradio coding-assistant demo.

#### Table of contents:

- [Prerequisites](#Prerequisites)
- [Download and Convert Model](#Download-and-Convert-Model)
- [Create Inference Pipeline](#Create-Inference-Pipeline)
    - [Select Inference Device](#Select-Inference-Device)
    - [Load OpenVINO Model](#Load-OpenVINO-Model)
    - [Run Text Generation](#Run-Text-Generation)
- [Interactive Demo](#Interactive-Demo)

### [quantization_comparison.ipynb](quantization_comparison.ipynb)

A companion notebook that compares FP16, INT8, and INT4 weight compression for Qwen2.5-Coder-7B-Instruct, to help you choose the best trade-off between model size, speed, and code quality. In this notebook, we will:
1. Convert Qwen2.5-Coder-7B-Instruct to FP16, INT8, and INT4 OpenVINO IR formats.
2. Compare resulting model sizes on disk.
3. Benchmark inference speed (tokens per second) for each format.
4. Evaluate generated code quality across multiple programming tasks for each format.

#### Table of contents:

- [Prerequisites](#Prerequisites)
- [Convert Models with Different Quantization](#Convert-Models)
- [Compare Model Sizes](#Compare-Model-Sizes)
- [Benchmark Inference Speed](#Benchmark-Speed)
- [Evaluate Code Quality](#Evaluate-Quality)
- [Summary and Recommendation](#Summary)

### Hardware Requirements

Model size on disk (and approximate RAM needed to run it) depends on the model variant and the weight format chosen at export time:

| Model | Weight format | Size on disk | Recommended RAM |
|-------|---------------|--------------|------------------|
| Qwen2.5-Coder-1.5B-Instruct | fp16 | ~3 GB | ~4 GB |
| Qwen2.5-Coder-1.5B-Instruct | int8 | ~1.6 GB | ~3 GB |
| Qwen2.5-Coder-1.5B-Instruct | int4 | ~1 GB | ~2 GB |
| Qwen2.5-Coder-3B-Instruct | fp16 | ~6 GB | ~8 GB |
| Qwen2.5-Coder-3B-Instruct | int8 | ~3.2 GB | ~5 GB |
| Qwen2.5-Coder-3B-Instruct | int4 | ~2 GB | ~4 GB |
| Qwen2.5-Coder-7B-Instruct | fp16 | ~14 GB | ~16 GB |
| Qwen2.5-Coder-7B-Instruct | int8 | ~7 GB | ~8 GB |
| Qwen2.5-Coder-7B-Instruct | int4 | ~4 GB | ~5 GB |

> Sizes for the 3B variant are approximate, linearly interpolated between the measured 1.5B and 7B figures used in the notebooks; actual footprint can vary slightly with the OpenVINO/NNCF version used for conversion.

* **CPU**: All variants run on CPU-only machines with INT4 or INT8 weights. FP16 is only practical with 16 GB+ RAM (7B) or a GPU.
* **GPU**: Optional but recommended for the 7B model at FP16/INT8 for faster generation; select the device in the [Select Inference Device](#Select-Inference-Device) step.
* For most users, **16 GB RAM + 8 GB GPU** is a good baseline, with **INT4** giving the best size/speed/quality trade-off (see [quantization_comparison.ipynb](quantization_comparison.ipynb) for the full comparison).

### Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/Qwen2.5-Coder/README.md" />
