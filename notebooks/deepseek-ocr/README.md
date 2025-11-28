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


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).