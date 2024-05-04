# Visual Content Search using MobileCLIP and OpenVINO™
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/289-mobileclip-video-search/289-mobileclip-video-search.ipynb)

![example.png](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/4e241f82-548e-41c2-b1f4-45b319d3e519)

Semantic visual content search is a machine learning task that uses either a text query or an input image to search a database of images (photo gallery, video) to find images that are semantically similar to the search query. 
Historically, building a robust search engine for images was difficult. One could search by features such as file name and image metadata, and use any context around an image (i.e. alt text or surrounding text if an image appears in a passage of text) to provide the richer searching feature. This was before the advent of neural networks that can identify semantically related images to a given user query.

[Contrastive Language-Image Pre-Training (CLIP)](https://arxiv.org/abs/2103.00020) models provide the means through which you can implement a semantic search engine with a few dozen lines of code. The CLIP model has been trained on millions of pairs of text and images, encoding semantics from images and text combined. Using CLIP, you can provide a text query and CLIP will return the images most related to the query.

In this tutorial, we consider how to use [MobileCLIP](https://arxiv.org/pdf/2311.17049.pdf) for implementing a visual content search engine for finding relevant frames in video

## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run PyTorch MobileCLIP  with OpenVINO. It also provides an interactive user interface for search frames in video that are the most relevant to text or image requests.
The tutorial consists of the following steps:


- Select model
- Prepare PyTorch model
- Run PyTorch model inference
- Convert PyTorch model to OpenVINO IR
- Run model inference with OpenVINO
- Launch interactive demo for 


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).