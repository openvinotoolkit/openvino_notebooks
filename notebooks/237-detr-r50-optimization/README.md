# Convert and Optimize DETR Resnet-50 with OpenVINO™

<p align="center">
    <img src=""/>
</p>

This tutorial explains how to convert and optimize the [DETR Resnet-50](https://github.com/facebookresearch/detr) PyTorch model with OpenVNO.


## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run and optimize PyTorch DETR R-50 with OpenVINO.

The tutorial consists of the following steps:
- Prepare the PyTorch model.
- Download and prepare the dataset.
- Validate the original model.
- Convert the PyTorch model to OpenVINO IR.
- Validate the converted model.
- Prepare and run NNCF post-training optimization pipeline.
- Compare accuracy of the FP32 and quantized models.
- Compare performance of the FP32 and quantized models.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).

