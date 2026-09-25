# Crowd Counting with CSRNet and OpenVINO™

This notebook demonstrates crowd counting and provides high-quality density maps using
[CSRNet](https://arxiv.org/abs/1802.10062) (*Dilated Convolutional Neural Networks
for Understanding the Highly Congested Scenes*, CVPR 2018) with OpenVINO.

The notebook does the following --

1. Sets up a dedicated virtual environment **`.csrnet-nb-venv`** and installs every dependency (torch/torchvision from the Intel XPU index, OpenVINO, NNCF, and the rest).
2. Loads a pretrained CSRNet model (PyTorch).
3. Converts it to **OpenVINO IR** at **FP32** and **FP16**.
4. Quantizes it to **INT8** with **NNCF** post-training quantization.
5. Runs inference on CPU / GPU / NPU and visualizes the predicted density map.
6. Evaluates the prediction with **count error, PSNR and SSIM**, and benchmarks latency per precision (PyTorch included as the reference row).

For illustration, the notebook uses a single test image from **ShanghaiTech Part A** dataset whose ground-truth count is **141**. The model weights pretrained on ShanghaiTech Part A and published by the CSRNet authors are downloaded from the Google Drive link provided in the authors' GitHub repository (https://github.com/leeyeehoo/CSRNet-pytorch).

## Notebook Contents

- Install dependencies into the active kernel (self-contained, with the Intel XPU torch build).
- Download the model.
- Preprocessing and building the ground-truth density map (geometry-adaptive Gaussian kernel) for the test image.
- Convert to OpenVINO IR (FP32/FP16) and quantize to INT8 with NNCF.
- Select a device, run inference, visualize density maps.
- Report GT count, estimate, PSNR and SSIM per precision, plus a latency benchmark (PyTorch as the reference row).

## Installation Instructions

This is a self-contained example that relies solely on its own code. It is recommended to run the notebook in a dedicated virtual environment — you only need a Jupyter server to start. The *Install dependencies* cell documents how to create the **`.csrnet-nb-venv`** environment and register it as a Jupyter kernel. For general environment setup, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/README.md" />
