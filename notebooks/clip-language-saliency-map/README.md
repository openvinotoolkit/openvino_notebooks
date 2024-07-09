# Language-Visual Saliency with CLIP and OpenVINO™

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/clip-language-saliency-map/README.md" />
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/clip-language-saliency-map/clip-language-saliency-map.ipynb)

The notebook will cover the following topics:

* Explanation of a _saliency map_ and how it can be used.
* Overview of the CLIP neural network and its usage in generating saliency maps.
* How to split a neural network into parts for separate inference.
* How to speed up inference with OpenVINO™ and asynchronous execution.

## Saliency Map

A saliency map is a visualization technique that highlights regions of interest in an image. For example, it can be used to [explain image classification predictions](https://arxiv.org/abs/2110.08288) for a particular label. Here is an example of a saliency map that we will get in this notebook:

<p align="center">
    <img width="80%" src="https://user-images.githubusercontent.com/29454499/218967961-9858efd5-fff2-4eb0-bde9-60852f4b31cb.JPG"/>
</p>

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
