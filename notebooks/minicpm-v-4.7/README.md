# Visual-language assistant with MiniCPM-V 4.7 and OpenVINO

[MiniCPM-V 4.7](https://huggingface.co/docs/transformers/main/model_doc/minicpmv4_7) adds canvas M-RoPE to the MiniCPM-V architecture so image slices retain their spatial positions and video frames retain their temporal order.

The [notebook](./minicpm-v-4.7.ipynb) downloads a preconverted OpenVINO model by default. Uncheck **Use preconverted model** to export the 1B dense or 35B-A3B MoE checkpoint from a local directory or Hugging Face model ID with Optimum Intel main instead. Inference uses the nightly OpenVINO GenAI `VLMPipeline` for text, single-image, multi-image, document, and video responses. INT8 is the default weight format; INT4 and FP16 are selectable. Thinking can be enabled per inference request. The optional Gradio 6.29.1+ image demo uses [gradio_helper.py](./gradio_helper.py) and has an examples table with image previews, questions, and thinking settings; its streamed thinking appears in a collapsible block.

The [official model collection](https://huggingface.co/collections/openbmb/minicpm-v-47) includes [`openbmb/MiniCPM-V-4.7-1B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) and [`openbmb/MiniCPM-V-4.7-35B-A3B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B). Preconverted models are available for [1B INT8](https://huggingface.co/OpenVINO/MiniCPM-V-4.7-1B-int8-ov), [1B INT4](https://huggingface.co/OpenVINO/MiniCPM-V-4.7-1B-int4-ov), [1B FP16](https://huggingface.co/OpenVINO/MiniCPM-V-4.7-1B-fp16-ov), [35B-A3B INT8](https://huggingface.co/OpenVINO/MiniCPM-V-4.7-35B-A3B-int8-ov), and [35B-A3B INT4](https://huggingface.co/OpenVINO/MiniCPM-V-4.7-35B-A3B-int4-ov). The selected model downloads into `OpenVINO-<checkpoint>-ov/<weight-format>`; rerunning the cell uses the Hugging Face cache to resume or update it. There is no published 35B-A3B FP16 conversion; uncheck the box to export it locally.

For local conversion only, an empty `Export source:` field uses the selected checkpoint's local directory if it exists; otherwise, it exports the published `openbmb/<checkpoint>` model. Enter a local directory or Hugging Face ID to override this choice. Exports go to `<model-name>-ov/<weight-format>` (for example, `MiniCPM-V-4.7-1B-ov/int8`), separately from downloaded conversions and local source checkpoints. The 35B-A3B download requires substantial disk space; conversion also requires RAM for the original weights.

After the first download or export, the last cell can be run on its own after a kernel restart. The helper discovers downloaded and exported models, caches sample images under `demo_examples`, and loads the selected model on the first request; select the model and device in the Gradio form. Keep `gradio_helper.py` alongside the notebook. If the model was loaded by the notebook earlier in the same kernel, the helper reuses it.

The demo starts locally. Only if Gradio reports that localhost is inaccessible does the launch retry with `share=True`, creating a public link accessible to anyone with its URL. For remote hosting, set `server_name` and `server_port` explicitly in the last cell.

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO and uses development builds. It may be fully supported and validated in the future.

The notebook uses [Optimum Intel](https://github.com/huggingface/optimum-intel/pull/2033) for export and [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai/pull/4591) for inference. Keep the exported model's visual downsampling configuration; do not override it with `4x`. The larger 35B-A3B MoE checkpoint needs considerably more resources; its INT4 GPU PagedAttention path requires [this OpenVINO fix](https://github.com/openvinotoolkit/openvino/pull/38631).

The notebook installs Optimum Intel from main, Transformers 5.18.0 as used in the [4.7 export PR](https://github.com/huggingface/optimum-intel/pull/2033), and nightly OpenVINO, Tokenizers, and GenAI. These development builds can change; restart the kernel after installation. A full model run on each device is still required to validate results.

## Installation Instructions

Use a dedicated Python environment with Jupyter installed and run the notebook cells in order. The notebook installs its own model dependencies. For Jupyter setup, see the [installation guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/minicpm-v-4.7/README.md" />
