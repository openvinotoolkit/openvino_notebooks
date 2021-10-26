# Quantize a Segmentation Model and Show Live Inference

## Kidney Segmentation with PyTorch Lightning and OpenVINO™ - Part 2

![kidney segmentation](https://user-images.githubusercontent.com/15709723/134784204-cf8f7800-b84c-47f5-a1d8-25a9afab88f8.gif)

## What's Inside

This Jupyter notebook tutorial is part of a series that shows how to train,
optimize, quantize and show live inference on a segmentation model with PyTorch
Lightning and OpenVINO.

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
* Change to the notebook directory: `cd notebooks/210-ct-scan-live-inference`
* Install the requirements with `pip install --upgrade pip && pip install -r requirements.txt`.
* Run the notebook by typing `jupyter lab` and doubleclicking on the notebook from the left sidebar.

> NOTE: This notebook needs an internet connection to download data. If you use a proxy server, please enable the proxy server in the terminal before typing `jupyter lab`.

> NOTE: To use the other notebooks in the openvino_notebooks repository, please follow the instructions in the main README at https://github.com/openvinotoolkit/openvino_notebooks/

## Other Tutorials in this series 

> NOTE: Links to these notebooks will be published soon

* Data preparation: Demonstrates how to convert 3D medical data in Nifty format to 2D images for 2D inference and visualization
* Training: Demonstrates how to train a kidney segmentation model with annotated slices of CT scans using PyTorch Lightning and export as ONNX
* Optimization and Quantization: Demonstrates how to optimize and quantize the model that was created by the previous notebook and deploy on CPU, iGPU or combination of both devices
