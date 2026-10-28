# Document OCR with MiniCPM-OCR and OpenVINO

[MiniCPM-OCR](https://huggingface.co/docs/transformers/main/model_doc/minicpm_ocr) is a document-understanding model from the MiniCPM-V family developed by [OpenBMB](https://github.com/OpenBMB). It keeps the MiniCPM-V 4.6 architecture, a SigLIP vision encoder with a window-attention merger in front of a Qwen3.5 language model, and specializes it for reading text-dense images such as scanned pages, screenshots, receipts and tables.

The model reads images with `4x` visual downsampling by default, which keeps 4× more visual tokens than the `16x` mode of general-purpose MiniCPM-V models, and can reason in a thinking block before answering.

More details about the model can be found in the [model card](https://huggingface.co/openbmb/MiniCPM-OCR-1.0), the [Transformers documentation](https://huggingface.co/docs/transformers/main/model_doc/minicpm_ocr) and the original [repo](https://github.com/OpenBMB/MiniCPM-V).

### Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Convert the model to OpenVINO IR and compress its weights using Optimum Intel CLI
- Load the model with `OVModelForVisualCausalLM` on CPU, GPU or, as a preview, with the language model on NPU
- Recognize the text of a receipt, convert a page to Markdown, read Chinese text and extract key information from a document
- Score OCR results with the MiniCPM-OCR Jev quality-decision model loaded with `OVModelForVisualSequenceClassification`
- Measure accuracy against the original PyTorch model and generation performance
- Launch an interactive Gradio demo with optional Jev verification

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO. It uses Optimum Intel from the main branch, Transformers 5.21.0 and nightly OpenVINO builds. It may be fully supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/minicpm-ocr/README.md" />
