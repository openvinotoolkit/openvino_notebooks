# Granite 4.0-1B text generation with OpenVINO

Run the dense, all-attention Granite 4.0-1B instruction model locally with OpenVINO GenAI. Export and compress its weights with Optimum Intel and NNCF, stream chat responses, summarize a short document, and call an offline temperature-conversion tool.

The [Granite 4.0-1B checkpoint](https://huggingface.co/ibm-granite/granite-4.0-1b)
is instruction tuned and licensed under Apache-2.0. Although its class is
`GraniteMoeHybridForCausalLM`, this checkpoint has only attention layers and no
experts; it is not `granite-4.0-h-1b`.

**Experimental:** the notebook uses unreleased OpenVINO fixes and a pinned
Optimum Intel PR. Full-model output quality has not yet been validated.

## Notebook contents

The tutorial consists of the following steps:

1. Install the pinned dependencies and select CPU or an Intel GPU.
2. Export a pinned source revision to OpenVINO IR with INT8, INT4 or FP16 weights.
3. Stream a chat response and pass conversation history with OpenVINO GenAI.
4. Summarize a short document without downloading an additional dataset.
5. Validate a generated tool call, run a permitted offline temperature
   conversion, and return the result to the model.
6. Launch a Gradio chat interface with bounded input and output tokens.

INT8 weight compression is selected by default. NPU is not offered because
support for this exact checkpoint has not been established. No OpenVINO-author
preconverted IR was found when the notebook was prepared.

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to the [Installation Guide](../../README.md).

The install cells pin Optimum Intel to commit
`c9c3b841cd74728fc14952e7ef0680be416d1833` of
[PR 2117](https://github.com/huggingface/optimum-intel/pull/2117) for GPU
activation scaling, and use matching OpenVINO/Tokenizers/GenAI nightly packages
from November 16, 2026 for the
[GPU runtime fix](https://github.com/openvinotoolkit/openvino/pull/39233) and
[position-ID transformation](https://github.com/openvinotoolkit/openvino/pull/39112).
Transformers 5.5.0, NNCF 3.4.0 and Gradio 6.32.0 are also pinned.

See the [notebook](./granite-4.0.ipynb) for the complete installation and examples.
The tool example never evaluates generated code or calls an external service.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/granite-4.0/README.md" />
