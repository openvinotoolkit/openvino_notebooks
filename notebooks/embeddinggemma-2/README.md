# Text Embedding with EmbeddingGemma 2 and OpenVINO

This tutorial shows how to run the text embedding pipeline of [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) with OpenVINO.

EmbeddingGemma 2 is an open multimodal embedding model from Google DeepMind. It maps text, code, images, video, and audio into a single 768-dimensional vector space. The full model has 740M parameters: a 270M text backbone plus modular vision and audio encoders.

## Notebook Contents

The notebook contains the following steps:

1. Install OpenVINO, Optimum Intel, Transformers, and Sentence Transformers.
2. Export the Sentence Transformers checkpoint to OpenVINO IR with Optimum Intel.
3. Select an OpenVINO device. `NPU` is included when the OpenVINO NPU plugin and driver are installed; the notebook reshapes the exported IR to static input shapes before NPU inference.
4. Run text embedding inference with `OVSentenceTransformer`.
5. Apply task instruction prefixes and Matryoshka dimension truncation.
6. Optionally compare the OpenVINO embeddings with the original PyTorch model.

> **Note:** this notebook covers the text embedding path. The modular vision and audio encoders are not exported here.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/embeddinggemma-2/README.md" />
