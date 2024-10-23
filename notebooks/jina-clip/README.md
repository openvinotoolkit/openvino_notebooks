# CLIP model with Jina CLIP and OpenVINO

This tutorial will show how to run CLIP model pipeline with [jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1) model and OpenVINO.

## Notebook Contents

[jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1) is a state-of-the-art English multimodal(text-image) embedding model introduced in the [paper](https://arxiv.org/abs/2405.20204). It bridges the gap between traditional text embedding models, which excel in text-to-text retrieval but are incapable of cross-modal tasks, and models that effectively align image and text embeddings but are not optimized for text-to-text retrieval. jina-clip-v1 offers robust performance in both domains. Its dual capability makes it an excellent tool for multimodal retrieval-augmented generation (MuRAG) applications, allowing seamless text-to-text and text-to-image searches within a single model. jina-clip-v1 can be used for a variety of multimodal applications, such as: image search by describing them in text, multimodal question answering, multimodal content generation. Jina AI has also provided the Embeddings API as an easy-to-use interface for working with jina-clip-v1 and their other embedding models. 

The notebook contains the following steps:
1. Download the model and instantiate the PyTorch model.
2. Convert model to OpenVINO IR, using the model conversion API.
3. Quantize the converted model with NNCF.
4. Launch interactive demo


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md)..

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/jina-clip/README.md" />
