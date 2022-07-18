# Quantize a Segmentation Model and Show Live Inference

![kidney segmentation animation](https://user-images.githubusercontent.com/77325899/154279555-aaa47111-c976-4e77-8d23-aac96f45872f.gif)

## Notebook Contents

This folder contains four notebooks that show how to train,
optimize, quantize and show live inference on a [MONAI](https://monai.io/) segmentation model with
[PyTorch Lightning](https://pytorchlightning.ai/) and OpenVINO:

1\. [Data Preparation for 2D Segmentation of 3D Medical Data](data-preparation-ct-scan.ipynb)

2\. [Train a 2D-UNet Medical Imaging Model with PyTorch Lightning](pytorch-monai-training.ipynb)

3a. [Convert and Quantize a UNet Model and Show Live Inference using POT](110-ct-segmentation-quantize.ipynb)

3b. [Convert and Quantize a UNet Model and Show Live Inference using NNCF](110-ct-segmentation-quantize-nncf.ipynb)

The main difference between the POT and NNCF quantization notebooks is that NNCF performs quantization within the PyTorch framework, while POT performs
quantization after the PyTorch model has been converted to OpenVINO IR format. There is a pre-trained model and a subset of the dataset provided for the quantization notebook, 
so it is not required to run the data preparation and training notebooks before running the quantization tutorial.

This quantization tutorial consists of the following steps:

* Converting an ONNX model to OpenVINO IR with [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).
* Quantizing a model with the [Post-Training Optimization Tool](https://docs.openvino.ai/latest/pot_compression_api_README.html) API in OpenVINO.
* Evaluating the F1 score metric of the original model and the quantized model.
* Benchmarking performance of the original model and the quantized model.
* Showing live inference with async API and MULTI plugin in OpenVINO.

In addition to the notebooks in this folder, the [Live Inference and Benchmark CT-scan data](../210-ct-scan-live-inference/210-ct-scan-live-inference.ipynb) demo notebook contains 
the live-inference part of the quantization tutorial. It includes a pre-quantized model.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
