# Visual-language assistant with MiniCPM-V 4.7 and OpenVINO

[MiniCPM-V 4.7](https://huggingface.co/docs/transformers/main/model_doc/minicpmv4_7) adds canvas M-RoPE to the MiniCPM-V architecture so image slices retain their spatial positions and video frames retain their temporal order.

The [notebook](./minicpm-v-4.7.ipynb) downloads a preconverted OpenVINO model by default. Clear the **Use preconverted model** checkbox to export the 1B dense or 35B-A3B MoE checkpoint from a local directory or Hugging Face model ID with the pinned Optimum Intel development revision instead. Inference uses the nightly OpenVINO GenAI `VLMPipeline` for text, single-image, multi-image, document, and video responses. INT8 is the default weight format; INT4 and FP16 are selectable. Thinking can be enabled per inference request. The optional Gradio 6.29.1+ image demo uses [gradio_helper.py](./gradio_helper.py) and has an examples table with image previews, questions, and thinking settings; its streamed thinking appears in a collapsible block.

Text, image, and non-thinking Gradio responses allow up to 512 new tokens; the video example allows up to 2048, and the Gradio thinking mode allows up to 1024. These are limits, not required output lengths.

The [official model collection](https://huggingface.co/collections/openbmb/minicpm-v-47) includes [`openbmb/MiniCPM-V-4.7-1B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) and [`openbmb/MiniCPM-V-4.7-35B-A3B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B). Preconverted models are available for [1B INT8](https://huggingface.co/OpenVINO/MiniCPM-V-4.7-1B-int8-ov), [1B INT4](https://huggingface.co/OpenVINO/MiniCPM-V-4.7-1B-int4-ov), [1B FP16](https://huggingface.co/OpenVINO/MiniCPM-V-4.7-1B-fp16-ov), [35B-A3B INT8](https://huggingface.co/OpenVINO/MiniCPM-V-4.7-35B-A3B-int8-ov), and [35B-A3B INT4](https://huggingface.co/OpenVINO/MiniCPM-V-4.7-35B-A3B-int4-ov). The selected model downloads into `OpenVINO-<checkpoint>-ov/<weight-format>`; rerunning the cell uses the Hugging Face cache to resume or update it. There is no published 35B-A3B FP16 conversion; clear the checkbox to export it locally.

For local conversion only, an empty `Export source:` field uses the selected checkpoint's local directory if it exists; otherwise, it exports the published `openbmb/<checkpoint>` model. Enter a local directory or Hugging Face ID to override this choice. Exports go to `<model-name>-ov/<weight-format>` (for example, `MiniCPM-V-4.7-1B-ov/int8`), separately from downloaded conversions and local source checkpoints. The 35B-A3B download requires substantial disk space; conversion also requires RAM for the original weights.

The [published 1B INT4 conversion](https://huggingface.co/OpenVINO/MiniCPM-V-4.7-1B-int4-ov/blob/7743dbc831ea3073795d4bbbf73490cdb0498808/openvino_config.json) uses group size 128 without calibration. A fresh canonical-ID local export uses [tuned INT4 defaults](https://github.com/huggingface/optimum-intel/pull/2056): asymmetric group size 64, ratio 1.0, and GSM8K scale estimation. Neither a downloaded IR nor a previously exported directory is updated automatically; use a new output directory for re-export.

An optional **DSpark speculative decoding** section exports a separate, correctly paired target and `-DSpark` draft: 1B INT8 + INT4 draft, or 35B-A3B INT4 + INT4 draft. Enable **Run DSpark example** to download, export, and load this additional pair; leave it disabled for the ordinary tutorial. This text-only example uses the selected checkpoint but not the export-source/weight/download controls, writes to `dspark-models`, and preserves the draft-free image/video examples and Gradio demo. It requires memory for an additional target and draft. CPU/GPU continuous batching, greedy decoding, one return sequence, and no prefix caching are used; NPU is not supported. See the [merged GenAI implementation](https://github.com/openvinotoolkit/openvino.genai/pull/4790). Draft counts do not establish an accuracy or speed improvement.

After the first download or export, the last cell can be run on its own after a kernel restart. The helper discovers downloaded and exported models, caches sample images under `demo_examples`, and loads the selected model on the first request; select the model and device in the Gradio form. Keep `gradio_helper.py` alongside the notebook. If the model was loaded by the notebook earlier in the same kernel, the helper reuses it.

The demo starts locally. Only if Gradio reports that localhost is inaccessible does the launch retry with `share=True`, creating a public link accessible to anyone with its URL. For remote hosting, set `server_name` and `server_port` explicitly in the last cell.

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO and uses development builds. It may be fully supported and validated in the future.

The notebook uses [Optimum Intel](https://github.com/huggingface/optimum-intel/pull/2033) for export and [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai/pull/4591) for inference. Keep the exported model's visual downsampling configuration; do not override it with `4x`. The larger 35B-A3B MoE checkpoint needs considerably more resources; its INT4 GPU PagedAttention path requires [this OpenVINO fix](https://github.com/openvinotoolkit/openvino/pull/38631).

Use Python 3.11–3.13. The notebook pins Optimum Intel to [`f09a56ac`](https://github.com/huggingface/optimum-intel/commit/f09a56ace88149cd341233ddfa2bc018bff7c079), Transformers to 5.18.0, NNCF to 3.4.0, OpenVINO to `2026.5.1.dev20261116`, and Tokenizers/GenAI to `2026.5.1.0.dev20261116`. These development versions include the new DSpark export/runtime contract and [GPU PagedAttention masking fix](https://github.com/openvinotoolkit/openvino/pull/38897); older preview drafts need re-export.

The exporter declares `transformers<5.16`, although MiniCPM-V 4.7 requires 5.18. As in the [upstream MiniCPM DSpark CI setup](https://github.com/openvinotoolkit/openvino.genai/blob/c6f9c59272851f89800754c63bf28bf365b5cd54/.github/workflows/test_matrices/windows/wheel_tests.yml), dependencies are installed explicitly and Optimum Intel is installed with `--no-deps`. `pip check` reports this known metadata conflict; the environment is not resolver-clean. Restart the kernel after installation. Full-model execution on each advertised device is still required to validate results.

## Installation Instructions

Use a dedicated Python environment with Jupyter installed and run the notebook cells in order. The notebook installs its own model dependencies. For Jupyter setup, see the [installation guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/minicpm-v-4.7/README.md" />
