# STEM Image Denoising with MCNN and OpenVINO™

[MCNN](https://github.com/fengwang/MCNN) (Multi-Resolution Convolutional Neural Network) is a deep model for denoising High-Angle Annular Dark-Field Scanning Transmission Electron Microscopy (HAADF-STEM) images. A STEM image acquired at low electron dose is dominated by salt-and-pepper noise that buries the real signal — the positions of atomic columns. MCNN learns to map the noisy image back to a clean reconstruction of the atomic structure.

The original model ships in Keras / TensorFlow 1.x, which no longer runs on modern Python. This notebook uses a maintained fork, [dhandhalyabhavik/MCNN-ov](https://github.com/dhandhalyabhavik/MCNN-ov), that re-implements the trained model in PyTorch (loading the original weights directly from the Keras H5 with `h5py`, no TensorFlow), and demonstrates converting it to OpenVINO IR, quantizing it to INT8, and comparing FP16 and INT8 on Intel CPU and GPU.

You can find more information about this model in the [research paper](https://doi.org/10.1038/s41598-020-62484-z) and the original GitHub [repository](https://github.com/fengwang/MCNN).

## Notebook Contents

This notebook demonstrates STEM image denoising with the MCNN model using OpenVINO.

The tutorial consists of the following steps:

- Prepare the PyTorch model from the fork's ported weights.
- Prepare data and run PyTorch inference (noisy to denoised).
- Convert the model to OpenVINO IR (FP16).
- Run OpenVINO inference on Intel CPU or GPU.
- Quantize the model to INT8 (weight-only) with NNCF.
- Compare the FP16 and INT8 models on performance and accuracy.

This model runs on Intel CPU and GPU. NPU is not used.

## Note on INT8 quantization

Full INT8 post-training quantization (weights and activations) collapses this
model: the sparse, high-dynamic-range activations of the 7-level U-Net saturate
the final sigmoid to a constant output. The notebook therefore uses weight-only
INT8 quantization (`nncf.compress_weights`), which keeps activations in floating
point. It is essentially lossless here (about 0.03 dB) and halves the model size
versus FP16.

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start. For details, please refer to [Installation Guide](../../README.md).
