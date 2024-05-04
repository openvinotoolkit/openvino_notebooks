# Convert and Optimize YOLOv9 with OpenVINO™

<p align="center">
    <img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/ae3a7653-eead-4c41-9cad-a7c95d3a4578"/>
</p>

YOLOv9 marks a significant advancement in real-time object detection, introducing groundbreaking techniques such as Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). This model demonstrates remarkable improvements in efficiency, accuracy, and adaptability, setting new benchmarks on the MS COCO dataset. More details about model can be found in [paper](https://arxiv.org/abs/2402.13616) and [original repository](https://github.com/WongKinYiu/yolov9).

## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to run and optimize PyTorch YOLO V9 with OpenVINO.

The tutorial consists of the following steps:

- Prepare PyTorch model
- Convert PyTorch model to OpenVINO IR
- Run model inference with OpenVINO
- Prepare and run optimization pipeline
- Compare performance of the FP32 and quantized models.
- Run optimized model inference on video


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
