# VLM NPU Notebook

This folder contains the `vlm-npu.ipynb` notebook for running vision-language models with OpenVINO. The models used in the notebook are specifically optimized to work on Intel NPU, though it may also work on CPU & GPU as well. This notebook is self-sufficient and install all the packages required to run the models within a virtual environment. The notebook downloads the models from HuggingFace (some models might require HF token), quantize and convert to OpenVINO IR format using optimum-cli and then pass an image and a prompt to generate the response using openvino-genai. 

The notebook is tested on Intel Core Ultra 3 (Panther Lake) NPU

## Prerequisites
- Create and activate a Python virtual environment.
- The notebook installs all required Python packages in its first setup cell.

## Usage
1. Open `vlm-npu.ipynb`.
2. Run the cells from top to bottom.
