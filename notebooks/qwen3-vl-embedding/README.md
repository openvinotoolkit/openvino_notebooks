# Multimodal Embedding and Reranker with Qwen3-VL and OpenVINO

The Qwen3-VL-Embedding and Qwen3-VL-Reranker model series are the latest additions to the Qwen family, built upon the powerful Qwen3-VL foundation model. Specifically designed for multimodal information retrieval and cross-modal understanding, this suite accepts diverse inputs including text, images, screenshots, and videos, as well as inputs containing a mixture of these modalities.

While the Embedding model generates high-dimensional vectors for broad applications like retrieval and clustering, the Reranker model is engineered to refine these results, establishing a comprehensive pipeline for state-of-the-art multimodal search.

<img src="https://model-demo.oss-cn-hangzhou.aliyuncs.com/Qwen3-VL-Embedding.png" width="500"/>

In this tutorial we consider how to convert and optimize Qwen3-VL Embedding and Reranker models using OpenVINO.

### Notebook Contents

The tutorial consists of the following steps:

- Prerequisites
- Select model
- Convert model using Optimum Intel
- Run OpenVINO model inference with Optimum-intel

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For further details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen3-vl-embedding/README.md" />
