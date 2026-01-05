# Benchmark: Qwen 2.5 (0.5B) INT4 Quantization on CPU

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/benchmark-qwen-2.5-int4/benchmark-qwen-2.5-int4.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/latest?filepath=notebooks%2Fbenchmark-qwen-2.5-int4%2Fbenchmark-qwen-2.5-int4.ipynb)

This notebook demonstrates the performance capabilities of OpenVINO™ by benchmarking the **Qwen 2.5 0.5B** model. It compares the latency of the standard PyTorch FP32 model against the OpenVINO INT4 quantized model.

## Notebook Contents

This tutorial covers:
1.  Downloading the Qwen 2.5 model from Hugging Face.
2.  Converting and Quantizing the model to INT4 using `optimum-cli`.
3.  Benchmarking inference speed (Latency) on CPU.

## Installation Instructions

This is a self-contained notebook. All requirements are installed within the first cell.
The notebook requires Python >= 3.8.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/benchmark-qwen-2.5-int4/README.md" />
