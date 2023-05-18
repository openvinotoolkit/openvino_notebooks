# Universal segmentation with OneFormer with OpenVINO™


This tutorial explains how to convert and run inference on the [OneFormer](https://huggingface.co/docs/transformers/model_doc/oneformer) HuggingFace model with OpenVNO.


## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run HuggingFace OneFormer with OpenVINO.

The tutorial consists of the following steps:
- Install required libraries
- Prepare the environment
- Load OneFormer fine-tuned on COCO for universal segmentation
- Convert PyTorch model to ONNX
- Convert ONNX model to OpenVINO IR
- Prepare the image
- Compile OpenVINO model
- Semantic segmentation
- Instance segmentation
- Panoptic segmentation