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

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/big-transfer-quantization/README.md" />
