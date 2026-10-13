# Video understanding with V-JEPA 2.1 and OpenVINO

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/vjepa-2.1/vjepa2-video-embeddings.ipynb)

[V-JEPA 2](https://ai.meta.com/vjepa/) (Video Joint-Embedding Predictive Architecture) is a self-supervised video model from Meta AI. Unlike a classifier, it is never trained to name what it sees. It is trained to **predict masked parts of a video in its own representation space** rather than in pixel space, which forces the representation to capture the structure that actually matters such as objects, surfaces and motion, while discarding unpredictable low-level detail.

The result is a general-purpose video encoder. It emits a grid of patch embeddings that can be reused for retrieval, segmentation, action recognition, robotic planning or motion analysis, usually with nothing more than a small head trained on top. **V-JEPA 2.1** refines this recipe with a distilled encoder family and 3D rotary position embeddings (RoPE) over the (time, height, width) token grid.

This tutorial takes the encoder end to end with OpenVINO:

* rebuild the encoder from the released checkpoint, in a form designed to map cleanly onto OpenVINO ops,
* convert it to OpenVINO Intermediate Representation and verify the conversion is numerically faithful,
* perform INT8 quantization.

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Load the V-JEPA 2.1 encoder checkpoint
- Prepare an input video clip
- Run the PyTorch model
- Convert the model to OpenVINO Intermediate Representation format
- Run OpenVINO model inference and verify the conversion
- Optimize the model with INT8 quantization
- Compare accuracy and size of the original and quantized models

## Installation Instructions

We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/vjepa-2.1/README.md" />
