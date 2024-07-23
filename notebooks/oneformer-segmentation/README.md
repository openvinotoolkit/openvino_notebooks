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


### Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/oneformer-segmentation/README.md" />
