# Document Parsing using DeepSeek-OCR and OpenVINO

DeepSeek-OCR, a VLM designed as a preliminary proof-of-concept for efficient vision-text compression.  DeepSeek-OCR consists of two components: DeepEncoder and DeepSeek3B-MoE-A570M as the decoder. Specifically, DeepEncoder serves as the core engine, designed to maintain low activations under high-resolution input while achieving high compression ratios to ensure an optimal and manageable number of vision tokens.

More details can be found in the [paper](https://arxiv.org/pdf/2510.18234), original [repository](https://github.com/deepseek-ai/DeepSeek-OCR) and [model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR).

In this tutorial we consider how to convert and run DeepSeek-OCR models using [OpenVINO](https://github.com/openvinotoolkit/openvino) and optimize it using [NNCF](https://github.com/openvinotoolkit/nncf).

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content.

<img width="1704" height="1125" alt="image" src="https://github.com/user-attachments/assets/46862d95-5e2e-4b0c-b5e1-55eebf2c86e5" />

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/deepseek-ocr/README.md" />
