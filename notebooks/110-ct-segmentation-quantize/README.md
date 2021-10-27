# Quantize a Segmentation Model and Show Live Inference

## Kidney Segmentation with PyTorch Lightning and OpenVINO™ - Part 3

![kidney segmentation](https://user-images.githubusercontent.com/15709723/134784204-cf8f7800-b84c-47f5-a1d8-25a9afab88f8.gif)

## What's Inside

This Jupyter notebook tutorial is part of a series that shows how to train,
optimize, quantize and show live inference on a segmentation model with PyTorch
Lightning and OpenVINO.

This third tutorial in the series shows how to:

- Convert an ONNX model to OpenVINO IR with [Model Optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html),
- Quantize a model with OpenVINO's [Post-Training Optimization Tool](https://docs.openvinotoolkit.org/latest/pot_compression_api_README.html) API. 
- Evaluate the F1 score metric of the original model and the quantized model
- Benchmark performance of the original model and the quantized model
- Show live inference with OpenVINO's async API and MULTI plugin



## Installation Instructions

For a minimum installation:

* Create a virtual environment, with `python -m venv openvino_env` (on Linux
  you may need to type `python3`) and activate it with
  `openvino_env\Scripts\activate` on Windows or `source
  openvino_env/bin/activate` on macOS or Linux.
* Clone the openvino_notebooks repository: `git clone
  https://github.com/openvinotoolkit/openvino_notebooks/`
* Change to the directory: `cd openvino_notebooks`
* Check out the live_inference branch `git checkout live_inference`
* Change to the notebook directory: `cd notebooks/110-ct-segmentation-quantize`
* Install the requirements with `pip install --upgrade pip && pip install -r requirements.txt`.
* Run the notebook by typing `jupyter lab` and doubleclicking on the notebook from the left sidebar.

> NOTE: This notebook needs an internet connection to download data. If you use a proxy server, please enable the proxy server in the terminal before typing `jupyter lab`.

> NOTE: To use the other notebooks in the openvino_notebooks repository, please follow the instructions in the main README at https://github.com/openvinotoolkit/openvino_notebooks/

## All Tutorials in this series 

- [Data Preparation for 2D Segmentation of 3D Medical Data](data-preparation-ct-scan.ipynb)
- Train a 2D-UNet Medical Imaging Model with PyTorch Lightning (Will be added soon)
- Convert and Quantize a UNet Model and Show Live Inference (This tutorial)
- [Live Inference and Benchmark CT-scan data](../210-ct-scan-live-inference/210-ct-scan-live-inference.ipynb)
