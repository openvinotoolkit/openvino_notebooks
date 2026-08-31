# RF-DETR Object Detection with OpenVINO

This tutorial demonstrates how to convert RF-DETR object-detection models from Hugging Face to OpenVINO IR and run inference with the Optimum Intel API.

The notebook provides a model selector for the Apache-2.0 RF-DETR Nano, Small, Medium, Base, and Large checkpoints. Nano is the default model for a faster first run.

## Notebook Contents

- Install the Optimum Intel RF-DETR feature branch with the shared `pip_install` helper.
- Select an RF-DETR model and an OpenVINO CPU or GPU device.
- Export and cache the model as OpenVINO IR.
- Run object detection and visualize bounding boxes on a sample image.
- Launch an interactive Gradio object-detection demo.

## Installation Instructions

Follow the [OpenVINO Notebooks installation guide](https://github.com/openvinotoolkit/openvino_notebooks#-installation-guide). Until RF-DETR support is merged upstream, the notebook installs it from the [`fix/rf-detr-followup-1843`](https://github.com/aleksandr-mokrov/optimum-intel/tree/fix/rf-detr-followup-1843) Optimum Intel feature branch.

The notebook installs Transformers 5.10.x after Optimum Intel because RF-DETR is not available in the Transformers version range currently declared by Optimum Intel.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/rf-detr-object-detection/README.md" />
