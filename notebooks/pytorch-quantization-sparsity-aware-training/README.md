# Optimizing PyTorch models with Neural Network Compression Framework of OpenVINO™ by 8-bit quantization.

This tutorial demonstrates how to use [NNCF](https://github.com/openvinotoolkit/nncf) 8-bit sparse quantization to optimize the 
[PyTorch](https://pytorch.org/) model for inference with [OpenVINO Toolkit](https://docs.openvino.ai/). 
For more advanced usage, refer to these [examples](https://github.com/openvinotoolkit/nncf/tree/develop/examples).

This notebook is based on 'ImageNet training in PyTorch' [example](https://github.com/pytorch/examples/blob/master/imagenet/main.py).
This notebook uses a [ResNet-50](https://arxiv.org/abs/1512.03385) model with the 
ImageNet dataset.

## Notebook Contents

This tutorial consists of the following steps:
* Transforming the original dense `FP32` model to sparse `INT8`
* Using fine-tuning to restore the accuracy.
* Exporting optimized and original models to OpenVINO
* Measuring and comparing the performance of the models.

## Installation Instructions

This is a self-contained example that relies solely on its own code and accompanying config.json file.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).


<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/pytorch-quantization-sparsity-aware-training/README.md" />
