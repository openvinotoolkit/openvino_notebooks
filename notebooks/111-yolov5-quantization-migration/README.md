# Migrate quantization from POT API to NNCF API
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/111-yolov5-quantization-migration/111-yolov5-quantization-migration.ipynb)

![Ultralytics YOLOv5 results](https://user-images.githubusercontent.com/44352144/177097174-cfe78939-e946-445e-9fce-d8897417ef8e.png)


This tutorial demonstrates how to migrate quantization pipeline written using the OpenVINO [Post-Training Optimization Tool (POT)](https://docs.openvino.ai/2023.0/pot_introduction.html) to [NNCF Post-Training Quantization API](https://docs.openvino.ai/nightly/basic_quantization_flow.html). This tutorial is based on  [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) model and additionally it compares model accuracy between the FP32 precision and quantized INT8 precision models and runs a demo of model inference based on sample code from [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) with the OpenVINO backend.


## Notebook Contents

The tutorial consists from the following parts:

1. Convert YOLOv5 model to OpenVINO IR.
2. Prepare dataset for quantization.
3. Configure quantization pipeline.
4. Perform model optimization.
5. Compare accuracy FP32 and INT8 models
6. Run model inference demo
7. Compare performance FP32 and INT8 models

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
