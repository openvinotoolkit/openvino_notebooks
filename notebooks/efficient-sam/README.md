# Object segmentations with EfficientSAM and OpenVINO

Segment Anything Model (SAM) has emerged as a powerful tool for numerous vision applications. A key component that drives the impressive performance for zero-shot transfer and high versatility is a super large Transformer model trained on the extensive high-quality SA-1B dataset. While beneficial, the huge computation cost of SAM model has limited its applications to wider real-world applications. To address this limitation, EfficientSAMs, light-weight SAM models that exhibit decent performance with largely reduced complexity, were proposed. The idea behind EfficientSAM is based on leveraging masked image pretraining, SAMI, which learns to reconstruct features from SAM image encoder for effective visual representation learning.

![overview.png](https://yformer.github.io/efficient-sam/EfficientSAM_files/overview.png)

More details about model can be found in [paper](https://arxiv.org/pdf/2312.00863.pdf), [model web page](https://yformer.github.io/efficient-sam/) and [original repository](https://github.com/yformer/EfficientSAM)

In this tutorial, we consider how to convert and run EfficientSAM using OpenVINO. We also demonstrate how to quantize model using [NNCF](https://github.com/openvinotoolkit/nncf)

The image below illustrates the result of the segmented area of the image by provided points

![example.png](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/15d0a23a-0550-43c6-9ca9-f689e772a79a)


### Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Load PyTorch model
- Run PyTorch model inference
- Convert PyTorch model to OpenVINO Intermediate Representation
- Run OpenVINO model inference
- Optimize OpenVINO model using [NNCF](https://github.com/openvinotoolkit/nncf)
- Launch interactive segmentation demo

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
