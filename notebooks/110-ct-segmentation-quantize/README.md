# Quantize a Segmentation Model and Show Live Inference

![kidney segmentation](https://user-images.githubusercontent.com/77325899/146994250-321fe29f-5829-491b-83a2-b8be73684c6b.gif)

## What's Inside

This folder contains three notebooks that show how to train,
optimize, quantize and show live inference on a [MONAI](https://monai.io/) segmentation model with
[PyTorch Lightning](https://pytorchlightning.ai/) and OpenVINO:

1. [Data Preparation for 2D Segmentation of 3D Medical Data](data-preparation-ct-scan.ipynb)
2. [Train a 2D-UNet Medical Imaging Model with PyTorch Lightning](pytorch-monai-training.ipynb)
3. [Convert and Quantize a UNet Model and Show Live Inference](110-ct-segmentation-quantize.ipynb)


We provided a pretrained model and a subset of the dataset for the quantization notebook, so it is not required to run the data preparation and training notebooks before running the quantization tutorial.

The quantization tutorial shows how to:

- Convert an ONNX model to OpenVINO IR with [Model Optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
- Quantize a model with OpenVINO's [Post-Training Optimization Tool](https://docs.openvinotoolkit.org/latest/pot_compression_api_README.html) API
- Evaluate the F1 score metric of the original model and the quantized model
- Benchmark performance of the original model and the quantized model
- Show live inference with OpenVINO's async API and MULTI plugin

In addition to the notebooks in this folder, the [Live Inference and Benchmark CT-scan data](../210-ct-scan-live-inference/210-ct-scan-live-inference.ipynb) demo notebook contains the live-inference part of the quantization tutorial. It includes a pre-quantized model.

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
