# Optimizing PyTorch models with Neural Network Compression Framework of OpenVINO by 8-bit quantization.

This tutorial demonstrates how to use [NNCF](https://github.com/openvinotoolkit/nncf) 8-bit quantization to optimize the 
[PyTorch](https://pytorch.org/) model for inference with [OpenVINO Toolkit](https://docs.openvinotoolkit.org/). 
For more advanced usage refer to these [examples](https://github.com/openvinotoolkit/nncf/tree/develop/examples)

This notebook is based on 'ImageNet training in PyTorch' [example](https://github.com/pytorch/examples/blob/master/imagenet/main.py).
To make downloading and training fast, we use [ResNet-18](https://arxiv.org/abs/1512.03385) model with 
[Tiny ImageNet](http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf) dataset.

It consists of the following steps:
- Transform the original FP32 model to INT8
- Use fine-tuning to restore the accuracy
- Export optimized and original models to ONNX and then to OpenVINO
- Measure and compare the performance of models
