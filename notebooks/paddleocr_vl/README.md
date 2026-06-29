# PaddleOCR-VL / PaddleOCR-VL-1.5 / PaddleOCR-VL-1.6 with OpenVINO™

<p align="center" width="100%">
    <img width="90%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/PaddleOCR-VL-1.5.png">
</p>

This notebook shows an end-to-end workflow for **PaddleOCR-VL → OpenVINO**:

- Select and download a pretrained model — PaddleOCR-VL-1.6, PaddleOCR-VL-1.5, or the base PaddleOCR-VL (they share the same architecture and preprocessing and differ only in their trained weights).
- Patch `modeling_paddleocr_vl.py` locally (for `trust_remote_code`).
- Convert/export the model to OpenVINO IR (optionally with INT4/INT8 weight compression).
- Validate the OpenVINO inference pipeline on an input image.

## Installation Instructions

This is a self-contained example that relies on the code in this folder.
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/paddleocr_vl/README.md" />