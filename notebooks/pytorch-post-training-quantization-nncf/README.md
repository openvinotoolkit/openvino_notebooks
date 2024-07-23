# Post-Training Quantization of PyTorch models with NNCF

This tutorial demonstrates how to use [NNCF](https://github.com/openvinotoolkit/nncf) 8-bit quantization in
post-training mode (without the fine-tuning pipeline) to optimize a [PyTorch](https://pytorch.org/) model
for high-speed inference via [OpenVINO Toolkit](https://docs.openvino.ai/). For more advanced NNCF
usage, refer to these [examples](https://github.com/openvinotoolkit/nncf/tree/develop/examples).

To speed up download and validation, this tutorial uses a pre-trained [ResNet-50](https://arxiv.org/abs/1512.03385)
model on the [Tiny ImageNet](http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf) dataset.

## Notebook contents

The tutorial consists of the following steps:

* Evaluating the original model.
* Transforming the original `FP32` model to `INT8`.
* Exporting optimized and original models to ONNX and then to OpenVINO IR.
* Comparing performance of the obtained `FP32` and `INT8` models.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/pytorch-post-training-quantization-nncf/README.md" />
