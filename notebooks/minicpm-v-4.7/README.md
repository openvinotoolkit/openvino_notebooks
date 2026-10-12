# Visual-language assistant with MiniCPM-V 4.7 and OpenVINO

[MiniCPM-V 4.7](https://huggingface.co/docs/transformers/main/model_doc/minicpmv4_7) adds canvas M-RoPE to the MiniCPM-V architecture so image slices retain their spatial positions and video frames retain their temporal order.

The [notebook](./minicpm-v-4.7.ipynb) uses Optimum Intel main to export either the 1B dense or 35B-A3B MoE checkpoint from a local directory or a Hugging Face model ID, then runs inference with the nightly OpenVINO GenAI `VLMPipeline`. It streams text, single-image, multi-image, document, and video responses. INT8 is the default weight format; INT4 and FP16 are selectable. Thinking can be enabled per inference request. The optional Gradio 6.29.1+ image demo uses [gradio_helper.py](./gradio_helper.py) and has an examples table with image previews, questions, and thinking settings; its streamed thinking appears in a collapsible block.

After the first export, the last cell can be run on its own after a kernel restart. The helper discovers exported models, caches sample images under `MiniCPM-V-4.7-ov/demo_examples`, and loads the selected model on the first request; select the model and device in the Gradio form. Keep `gradio_helper.py` alongside the notebook. If the model was loaded by the notebook earlier in the same kernel, the helper reuses it.

The demo starts locally. Only if Gradio reports that localhost is inaccessible does the launch retry with `share=True`, creating a public link accessible to anyone with its URL. For remote hosting, set `server_name` and `server_port` explicitly in the last cell.

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO and is using a custom branch of optimum-intel. It may be fully supported and validated in the future.

The notebook uses [Optimum Intel's merged 4.7 export support](https://github.com/huggingface/optimum-intel/pull/2033) and [GenAI's merged 4.7 inference support](https://github.com/openvinotoolkit/openvino.genai/pull/4591). Keep the exported model's visual downsampling configuration; do not override it with `4x`. The larger 35B-A3B MoE checkpoint needs considerably more resources; its INT4 GPU PagedAttention path requires [this OpenVINO fix](https://github.com/openvinotoolkit/openvino/pull/38631).

The notebook installs Optimum Intel from main, Transformers 5.18.0 as used in the [4.7 export PR](https://github.com/huggingface/optimum-intel/pull/2033), and nightly OpenVINO, Tokenizers, and GenAI. These development builds can change; restart the kernel after installation. A full model run on each device is still required to validate results.

## Installation Instructions

Use a dedicated Python environment with Jupyter installed and run the notebook cells in order. The notebook installs its own model dependencies. For Jupyter setup, see the [installation guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/minicpm-v-4.7/README.md" />
