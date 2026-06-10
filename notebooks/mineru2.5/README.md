# Document parsing with MinerU 2.5 and OpenVINO

[MinerU 2.5](https://github.com/opendatalab/MinerU) is a state-of-the-art document parsing engine maintained by [OpenDataLab](https://opendatalab.com/). Its 2.5 generation uses a single 1.2 B parameter vision–language model — [`opendatalab/MinerU2.5-Pro-2604-1.2B`](https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B) — that converts page images into structured Markdown / JSON in two prompted steps:

1. **Layout detection** — the whole page is sent to the model with the prompt `Layout Detection:`. The model emits special `<|box_start|>… <|box_end|><|ref_start|>type<|ref_end|>` tokens that describe every region's bounding box and type (text, title, table, formula, image, chart, …).
2. **Per-region content recognition** — every region is cropped and sent back with a region-specific prompt (`Text Recognition:`, `Table Recognition:`, `Formula Recognition:`, `Image Analysis:`).

A small post-processing layer fixes equation delimiters, normalises tables to HTML/OTSL, merges truncated paragraphs and finally renders Markdown via `json2md`.

The model is a Qwen2-VL architecture under the hood, so it can be exported to OpenVINO IR exactly like in the [`qwen2-vl`](../qwen2-vl/qwen2-vl.ipynb) notebook. In this tutorial we:

* convert and INT4 weight-compress `MinerU2.5-Pro-2604-1.2B` with [Optimum Intel](https://github.com/huggingface/optimum-intel),
* run inference with [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)'s `VLMPipeline`,
* drive the MinerU two-step pipeline through a small `OVMinerUClient` wrapper (in [`ov_mineru_helper.py`](ov_mineru_helper.py)) that reuses the official post-processing utilities from `mineru-vl-utils`,
* and expose the whole thing as a tiny Gradio document-to-Markdown demo.

## Notebook contents
The tutorial consists of the following steps:

- Install requirements
- Convert and optimize the model with `optimum-cli`
- Build an `OVMinerUClient` around `openvino_genai.VLMPipeline`
- Parse a sample document image and visualise the detected layout
- Convert a multi-page PDF to Markdown
- Launch an interactive Gradio demo

The demo accepts both PDF files (rasterised on the fly with [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)) and standalone document images, and renders the Markdown side by side with the source.

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/mineru2.5/README.md" />
