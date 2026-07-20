# Document Parsing using Unlimited-OCR and OpenVINO

**Unlimited-OCR** is a vision-language model (VLM) from Baidu designed for efficient, high-resolution document understanding and optical character recognition (OCR). It pairs a deep dual vision encoder (a SAM ViT-B encoder together with a CLIP-L encoder) and a linear projector with a compact DeepSeek-V2 **Mixture-of-Experts** decoder (12 layers, 64 routed + 2 shared experts, top-6 routing). To keep the decoder fast on long documents, every decoder layer uses **sliding-window attention** (window = 128) during generation while keeping all the image/prompt (prefill) tokens visible. Dynamic image tiling ("Gundam" mode) splits high-resolution pages into a variable number of tiles based on aspect ratio, so the model can read documents of (almost) unlimited size.

More details can be found in the original [model card](https://huggingface.co/baidu/Unlimited-OCR).

---

In this tutorial we consider how to convert and run Unlimited-OCR using [OpenVINO](https://github.com/openvinotoolkit/openvino) and optimize it using [NNCF](https://github.com/openvinotoolkit/nncf).

## Notebook contents
The tutorial consists of the following steps:

- Install requirements
- Download the original model from the Hugging Face Hub
- Convert and optimize each weighted sub-model to OpenVINO IR
  - text embeddings
  - vision encoder (SAM + CLIP + projector, two fixed-resolution IRs: 1024 global view and 640 crop tiles)
  - DeepSeek-V2 MoE language model (stateful, with faithful sliding-window attention)
- Run OpenVINO model inference (CPU / GPU)
- Launch an interactive Gradio demo

In this demonstration, you'll create an interactive document-parsing tool that converts images and PDFs to markdown, extracts raw text, and locates content with grounding bounding boxes.

### How the model is adapted for OpenVINO

OpenVINO compiles a *static* computation graph, so the parts of the original pipeline that rely on data-dependent control flow are handled outside the traced graph:

- **Vision encoders** are exported at two fixed square resolutions — one IR for the 1024² global view and one for the 640² crop tiles — each with a dynamic batch dimension for the variable number of tiles. Two IRs are needed because the SAM/CLIP positional-embedding interpolation is resolution dependent and cannot be traced as a single dynamic-resolution graph.
- The **MoE expert routing** is replaced with a statically-traceable loop over all experts (one-hot mask + `index_add_`), reproducing the original gate numerics.
- The **sliding-window attention** ring buffer is replaced with an explicit additive attention mask built in the inference wrapper: each query attends to all prefill tokens plus the last 128 generated tokens — numerically equivalent to the original, but expressible as a static graph with a growing stateful KV cache.
- Dynamic image tiling and the image-token/feature merge (`masked_scatter`) stay in Python, around the compiled sub-models.

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO. It may be fully supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/unlimited-ocr/README.md" />
