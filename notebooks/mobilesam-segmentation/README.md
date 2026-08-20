# Prompt-based image segmentation with MobileSAM and OpenVINO

[MobileSAM](https://github.com/ChaoningZhang/MobileSAM) replaces the Segment Anything image encoder with a lightweight <spell>TinyViT</spell> encoder while retaining prompt-based mask decoding. This tutorial converts the official PyTorch `vit_t` checkpoint directly to OpenVINO Intermediate Representation (IR), without an ONNX intermediate, and runs prompt-based image segmentation. For architecture and training details, see the [MobileSAM paper](https://arxiv.org/abs/2306.14289).

The model and source code are available under the Apache License 2.0. The inference example reuses the `coco_bike.jpg` sample hosted by OpenVINO Notebooks.

![MobileSAM point and box prompt segmentation](file.png)

## Notebook Contents

The notebook demonstrates how to:

1. Download a pinned revision of the official MobileSAM source and checkpoint.
2. Prepare representative inputs using MobileSAM preprocessing.
3. Convert the image encoder and prompt decoder directly from PyTorch with `openvino.convert_model`.
4. Save compressed FP16 OpenVINO IR models with stable tensor names and dynamic prompt dimensions.
5. Validate encoder embeddings, decoder outputs, and final masks against PyTorch.
6. Compile the models for a selected OpenVINO device.
7. Run point- and box-prompt inference on a shared OpenVINO Notebooks sample image.
8. Use a Gradio click-to-segment demo with cached image embeddings.
9. Measure device-dependent latency and model artifact size.

## Installation Instructions

This is a self-contained example that installs its Python dependencies and downloads all required assets at runtime. A Python 3.10–3.13 virtual environment is recommended. See the repository [Installation Guide](../../README.md#-installation-guide).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/mobilesam-segmentation/README.md" />
