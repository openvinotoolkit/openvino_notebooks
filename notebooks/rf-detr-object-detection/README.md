# RF-DETR Object Detection with OpenVINO

This tutorial demonstrates how to export official RF-DETR object-detection checkpoints to OpenVINO IR and run inference with the native OpenVINO API.

The notebook provides a model selector for the Apache-2.0 RF-DETR Nano, Small, Medium, Base, and Large checkpoints. Nano is the default model for a faster first run.

## Notebook Contents

- Install `rfdetr[openvino]` with the shared `pip_install` helper.
- Select an RF-DETR model and an OpenVINO CPU or GPU device.
- Export and cache the model as OpenVINO IR with FP16-compressed weights using the official RF-DETR exporter.
- Run object detection and visualize bounding boxes on a sample image.
- Launch an interactive Gradio object-detection demo.

## Installation Instructions

Follow the [OpenVINO Notebooks installation guide](https://github.com/openvinotoolkit/openvino_notebooks#-installation-guide). The notebook installs the official `rfdetr` package (version 1.11.0) with its OpenVINO export support, OpenVINO 2026.4 or newer, and Gradio 6.28.0. Model weights are downloaded automatically by `rfdetr` and are separate from the Hugging Face checkpoints used previously.

The exported IR is loaded through `openvino.Core` and run with preprocessing and postprocessing matching the RF-DETR object-detection pipeline.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/rf-detr-object-detection/README.md" />
