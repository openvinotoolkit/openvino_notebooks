# Congested Scene Recognition with CSRNet and OpenVINO™

This notebook demonstrates crowd counting as well as provide high-quality density maps using 
[CSRNet](https://arxiv.org/abs/1802.10062) (*Dilated Convolutional Neural Networks
for Understanding the Highly Congested Scenes*, CVPR 2018) with OpenVINO.

The notebook does the following -- 

1. Loads a pretrained CSRNet model (PyTorch).
2. Converts it to **OpenVINO IR** at **FP32** and **FP16**.
3. Quantizes it to **INT8** with **NNCF** post-training quantization.
4. Runs inference on CPU / GPU / NPU and visualizes the predicted density map.
5. Evaluates the prediction with **count error, PSNR and SSIM**.

For illustration, the notebook uses a single test image from **ShanghaiTech Part A** dataset whose ground-truth count is **141**. The model weights pretrained on ShanghaiTech Part A and published by the CSRNet authors are downloaded from the Google Drive link provided in the authors' GitHub repository (https://github.com/leeyeehoo/CSRNet-pytorch).

## Notebook Contents

- Installation of prerequisites.
- Download the model.
- Preprocessing and building the ground-truth density map (geometry-adaptive Gaussian kernel) for the test image.
- Convert to OpenVINO IR (FP32/FP16) and quantize to INT8 with NNCF.
- Select a device, run inference, visualize density maps.
- Report GT count, estimate, PSNR and SSIM per precision, plus a latency benchmark.

## Installation Instructions

This is a self-contained example. The first code cell installs the required packages. If you have them already, run through the rest of the cells.