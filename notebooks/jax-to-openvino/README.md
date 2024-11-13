# Convert a JAX Model to OpenVINO™ IR

[JAX](https://jax.readthedocs.io/en/latest) is a Python library for accelerator-oriented array computation and program transformation, designed for high-performance numerical computing and large-scale machine learning.
JAX provides a familiar NumPy-style API for ease of adoption by researchers and engineers.


In this tutorial we will show how to convert JAX [ViT](https://github.com/google-research/vision_transformer?tab=readme-ov-file#available-vit-models) and [Mixer](https://github.com/google-research/vision_transformer?tab=readme-ov-file#mlp-mixer) models in OpenVINO format.


<details>
  <summary><b>Click here for more detailed information about the models </b></summary>

### Vision Transformer
<img src="https://github.com/google-research/vision_transformer/blob/main/vit_figure.png?raw=true" width="800"> 

Overview of the model: authors split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. In order to perform classification, authors use the standard approach of adding an extra learnable "classification token" to the sequence.

### MLP-Mixer
<img src="https://github.com/google-research/vision_transformer/blob/main/mixer_figure.png?raw=true" width="800"> 

MLP-Mixer (Mixer for short) consists of per-patch linear embeddings, Mixer layers, and a classifier head. Mixer layers contain one token-mixing MLP and one channel-mixing MLP, each consisting of two fully-connected layers and a GELU nonlinearity. Other components include: skip-connections, dropout, and linear classifier head.

</details>


## Notebook contents
- Prerequisites
- Load and run the original model and a sample
- Infer the original model
- Convert the model to OpenVINO IR
- Compiling the model
- Run OpenVINO model inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/jax-to-openvino/README.md" />
