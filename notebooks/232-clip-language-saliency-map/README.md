# Language-Visual Saliency with CLIP and OpenVINO™

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

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
