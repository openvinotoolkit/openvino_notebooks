# PaddleOCR-VL with OpenVINO™

<p align="center" width="100%">
    <img width="90%" src="https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/paddleocrvl.png">
</p>

This notebook shows an end-to-end workflow for **PaddleOCR-VL → OpenVINO**:

- Download the pretrained PaddleOCR-VL model.
- Patch `modeling_paddleocr_vl.py` locally (for `trust_remote_code`).
- Convert/export the model to OpenVINO IR (optionally with INT4/INT8 weight compression).
- Validate the OpenVINO inference pipeline on an input image.

## Installation Instructions

This is a self-contained example that relies on the code in this folder.
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
