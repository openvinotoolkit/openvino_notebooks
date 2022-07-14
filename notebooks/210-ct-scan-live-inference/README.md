# Live Inference and Benchmark CT-scan Data with OpenVINO™

## Kidney Segmentation with PyTorch Lightning and OpenVINO - Part 4

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F210-ct-scan-live-inference%2F210-ct-scan-live-inference.ipynb)

![kidney segmentation animation](https://user-images.githubusercontent.com/77325899/154279555-aaa47111-c976-4e77-8d23-aac96f45872f.gif)

## Notebook Contents

This tutorial is a part of the series that shows how to train,
optimize, quantize and show live inference on a segmentation model with PyTorch
Lightning and OpenVINO.

This is the final part that shows benchmarking results and live inference on
the quantized segmentation model. You will see real-time segmentation of kidney
CT scans running on a CPU, iGPU, or combining both devices for higher
throughput. The processed frames are 3D scans that are shown as individual
slices. The visualization slides through the slices with detected kidneys
overlayed in red.  A pre-trained and quantized model is provided, so running the
previous tutorials in the series is not required.

## Installation Instructions

For a minimum installation:

* Create a virtual environment, with `python -m venv openvino_env` (on Linux
  you may need to type `python3`) and activate it with
  `openvino_env\Scripts\activate` on Windows or `source openvino_env/bin/activate` on macOS or Linux.
* Clone the openvino_notebooks repository: `git clone
  https://github.com/openvinotoolkit/openvino_notebooks/`
* Change to the directory: `cd openvino_notebooks`
* Check out the live_inference branch `git checkout live_inference`
* Change to the notebook directory: `cd notebooks/210-ct-scan-live-inference`
* Install the requirements with `pip install --upgrade pip && pip install -r requirements.txt`.
* Run the notebook by typing `jupyter lab` and doubleclicking on the notebook from the left sidebar.

> NOTE: This notebook needs an Internet connection to download data. If you use a proxy server, enable the proxy server in the terminal before typing `jupyter lab`.

> NOTE: To use other notebooks in the *openvino_notebooks* repository, follow the instructions in the [Installation Guide](../../README.md)

## All notebooks in this series:

- [Data Preparation for 2D Segmentation of 3D Medical Data](../110-ct-segmentation-quantize/data-preparation-ct-scan.ipynb)
- [Train a 2D-UNet Medical Imaging Model with PyTorch Lightning](../110-ct-segmentation-quantize/pytorch-monai-training.ipynb) 
- [Convert and Quantize a UNet Model and Show Live Inference](../110-ct-segmentation-quantize/110-ct-segmentation-quantize.ipynb)
- Live Inference and Benchmark CT-scan data (this notebook)

