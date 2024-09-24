# Structure Extraction with NuExtract and OpenVINO

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/nuextract-structure-extraction/nuextract-structure-extraction.ipynb)

[NuExtract](https://huggingface.co/numind/NuExtract) model is a text-to-JSON Large Language Model (LLM) that allows to extract arbitrarily complex information from text and turns it into structured data.

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites
- Download and convert the model from a public source using the [OpenVINO integration with Hugging Face Optimum](https://huggingface.co/blog/openvino)
- Compress model weights to INT8 and INT4 with [OpenVINO NNCF](https://github.com/openvinotoolkit/nncf)
- Create a structure extraction inference pipeline with [Generate API](https://github.com/openvinotoolkit/openvino.genai)
- Launch interactive Gradio demo with structure extraction pipeline

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/nuextract-structure-extraction/README.md" />
