# Streaming Content Moderation with Qwen3Guard and OpenVINO

Use [Qwen3Guard-Stream-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B) for text moderation with OpenVINO. Unlike a generative guard, this Qwen3-based model predicts three risk levels and separate input/output categories through four token-classification heads. Both the guard and the accompanying [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) generator use Apache-2.0 licenses.

![Qwen3Guard](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3Guard/Qwen3Guard_logo.png)

&#9888;&#65039; **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO and is using a custom branch of optimum-intel. It may be fully supported and validated in the future.

This notebook pins the [Optimum Intel moderation PR](https://github.com/huggingface/optimum-intel/pull/2039), which is not yet merged. It uses Optimum Intel because a released OpenVINO GenAI pipeline for this four-head classifier is not available. The device selector offers CPU and available Intel GPUs; the separate [NPUW support proposal](https://github.com/openvinotoolkit/openvino/pull/39353) is not included.

## Notebook contents

The tutorial consists of the following steps:

1. Install the pinned export and runtime dependencies.
2. Export the guard with KV-cache state and FP16 weights.
3. Inspect the four classification heads and moderate a user prompt.
4. Visualize token-by-token predictions over a fixed response, preserving conversation state.
5. Export Qwen3-0.6B and generate with prompt and response moderation.
6. Explore all three modes in Gradio.

The fixed-response example is a simulated stream, not text generation. Guarded generation blocks `Unsafe`, reports `Controversial`, and holds generated tokens until their chunk has been checked. Predictions can be wrong and do not replace an application's safety policy or evaluation. CI uses local tiny random models to exercise the APIs; their labels are not safety-quality evidence.

Guard inference uses FP32 computation. On CPU, it also uses an FP32 KV cache and disables dynamic activation quantization to keep incremental and full-prefix predictions consistent. INT8 weight compression is not offered because it changed a moderation decision in a small comparison against the original FP32 model.

## Installation Instructions

This is a self-contained example that relies solely on its own code.

Use Python 3.11 or newer and start with a Jupyter environment. The notebook creates its own dependency environment, pins Transformers 4.57.6 for the model's custom code, and downloads the two checkpoints only when their export cells run. Download and run all files in this directory together.
For details, please refer to the [Installation Guide](../../README.md).

Model details and safety taxonomy: [model card](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B), [Qwen3Guard project](https://github.com/QwenLM/Qwen3Guard). Export API: [Optimum Intel documentation](https://huggingface.co/docs/optimum-intel/openvino/export).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen3guard-stream/README.md" />
