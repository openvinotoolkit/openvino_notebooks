# Big Transfer Image Classification Model Quantization with NNCF in OpenVINO™

This tutorial demonstrates how to apply 'INT8' quantization to the [Big Transfer](https://tfhub.dev/google/bit/m-r50x1) Image Classification model. Here we demonstrate fine-tuning the model, OpenVINO optimization and followed by INT8 quantization processing with [NNCF](https://github.com/openvinotoolkit/nncf/).

## Notebook Contents

This tutorial consists of the following steps:
- Prepare Dataset.
- Plotting data samples.
- Model fine-tuning.
- Perform model optimization (IR) step.
- Compute model accuracy of the TF model.
- Compute model accuracy of the optimized model.
- Run nncf.Quantize for getting an Optimized model.
- Compute model accuracy of the quantized model.
- Compare model accuracy of the optimized and quantized models.
- Compare inference results on one picture 

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md#-installation-guide)