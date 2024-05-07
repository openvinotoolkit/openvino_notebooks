# Image generation using Latent Consistency Model and ONNX Runtime with OpenVINO Execution Provider 

LCMs: The next generation of generative models after Latent Diffusion Models (LDMs). 
Latent Diffusion models (LDMs) have achieved remarkable results in synthesizing high-resolution images. However, the iterative sampling is computationally intensive and leads to slow generation.

**Input text:** tree with lightning in the background, 8k

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/277367065-13a8f622-8ea7-4d12-b3f8-241d4499305e.png"/>
</p>

### Notebook Contents

This [notebook](./Latent Consistency Models with ONNX and OpenVINO Execution Provider.ipynb) demonstrates how to  run [LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) using ONNX Runtime and OpenVINO Execution Provider on iGPU of AI PC 



## Installation Instructions
We recommend to use virtual environment. To create it use python -m venv <virtual-environment-name>
To activate the virtual environment use \<virtual-environment-name>\Scripts\activate
Install onnxruntime-openvino 1.18.0 : pip install onnxruntime_openvino-1.18.0-cp311-cp311-win_amd64.whl 
Install OpenVINO 2024.1 on Windows from an Archive File as descreibed here https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-windows.html 
Initialize openvino (e.g., using .\w_openvino_toolkit_windows_2024.1.0.dev20240405_x86_64\w_openvino_toolkit_windows_2024.1.0.dev20240405_x86_64\setupvars.bat)
Now you only need a Jupyter server to start.
All other dependancies are installed in the notebook itself
For details, please refer to [Installation Guide](../../README.md).




