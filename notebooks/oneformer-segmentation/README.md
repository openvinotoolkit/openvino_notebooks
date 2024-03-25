# Universal segmentation with OneFormer with OpenVINO™


This tutorial explains how to convert and run inference on the [OneFormer](https://huggingface.co/docs/transformers/model_doc/oneformer) HuggingFace model with OpenVINO. Additionally, [NNCF](https://github.com/openvinotoolkit/nncf/) quantization is applied to improve OneFormer segmentation speed.


## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run HuggingFace OneFormer with OpenVINO and quantize it with [NNCF](https://github.com/openvinotoolkit/nncf/).

The tutorial consists of the following steps:
- Install required libraries
- Prepare the environment
- Load OneFormer fine-tuned on COCO for universal segmentation
- Convert the model to OpenVINO IR format
- Select inference device
- Choose a segmentation task
- Inference
- Quantization
  - Preparing calibration dataset
  - Run quantization
  - Compare model size and performance
- Interactive demo