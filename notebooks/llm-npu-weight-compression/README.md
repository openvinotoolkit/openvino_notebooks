# Compare LLM weight compression formats on NPU with OpenVINO™ GenAI

Weight compression makes a large language model (LLM) small enough for a laptop, and the format to use depends on the device. This tutorial shows how to check, on the device you will use, whether a compressed model still answers like the original one.

## Notebook Contents

The tutorial compares several weight formats of [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), or optionally [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), on NPU, CPU or GPU. It consists of the following steps:

1. Generate reference answers to 20 short questions with the original model in Hugging Face Transformers.
2. Prepare the weight formats: the 4-bit formats that the [OpenVINO GenAI guide for NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html) describes, INT8 and FP16. The formats are exported with `optimum-cli` or downloaded from the Hugging Face Hub.
3. Run each format with the OpenVINO GenAI `LLMPipeline` on the selected device and count the answers that agree with the reference.
4. Measure the generation speed of each format.
5. Ask your own questions to all formats side by side in an interactive demo.

The models are released under the Apache 2.0 license.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/llm-npu-weight-compression/README.md" />
