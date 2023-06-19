# Accelerate Inference of Sparse Transformer Models with OpenVINO™ and 4th Gen Intel&reg; Xeon&reg; Scalable Processors
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/116-sparsity-optimization/116-sparsity-optimization.ipynb)

This tutorial demonstrates how to improve performance of sparse Transformer models with [OpenVINO](https://docs.openvino.ai/) on 4th Gen Intel&reg; Xeon&reg; Scalable processors. It uses a pre-trained model from the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library and shows how to convert it to the OpenVINO™ IR format and run inference on a CPU, using a dedicated runtime option that enables sparsity optimizations. It also demonstrates how to get more performance stacking sparsity with 8-bit quantization. To simplify the user experience, the [Hugging Face Optimum](https://huggingface.co/docs/optimum) library is used to convert the model to the OpenVINO™ IR format and quantize it using [Neural Network Compression Framework](https://github.com/openvinotoolkit/nncf).

>**NOTE**: This tutorial requires OpenVINO 2022.3 or newer and 4th Gen Intel&reg; Xeon&reg; Scalable processor that can be acquired on [Amazon Web Services (AWS)](https://aws.amazon.com/ec2/instance-types/r7iz/).

## Notebook Contents

The tutorial consists of the following steps:

- Download and quantize sparse the public BERT model, using OpenVINO integration with Hugging Face Optimum.
- Compare sparse 8-bit vs. dense 8-bit inference performance.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
