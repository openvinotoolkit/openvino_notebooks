# Image generation using Latent Consistency Model and ONNX Runtime with OpenVINO Execution Provider 

LCMs: The next generation of generative models after Latent Diffusion Models (LDMs). 
Latent Diffusion models (LDMs) have achieved remarkable results in synthesizing high-resolution images. However, the iterative sampling is computationally intensive and leads to slow generation.

**Input text:** tree with lightning in the background, 8k

<p align="center">
    <img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/105707993/73cb12e3-152d-463a-bb06-5ea0ddedc6d6"/>
</p>

### Notebook Contents

This notebook demonstrates how to  run [LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) using ONNX Runtime and OpenVINO Execution Provider on iGPU of AI PC 



## Installation Instructions
- Create a virtual environment using 
  ```sh  
  python -m venv <venv-name>
  ```
- To activate the virtual environment use
  ```sh
  \<venv-name>\Scripts\activate
  ```
- Install onnxruntime-openvino 1.18.0 :
  ```sh
  pip install onnxruntime_openvino-1.18.0-cp311-cp311-win_amd64.whl
  ```
- Install OpenVINO 2024.1 on Windows from an Archive File as described here https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-windows.html 
- In Command Prompt initialize OpenVINO, for example using
  ```sh
  "C:\Program Files (x86)\Intel\openvino_2024\setupvars.bat"
  ```
- Now you only need a Jupyter server to start.
- All other dependencies are installed in the notebook itself
- For details, please refer to [Installation Guide](../../README.md#-installation-guide).


## How to build onnxruntime_openvino 
- onnxruntime_openvino-1.18 is used along with OpenVINO 2024.1 

- Ensure Visual Studio is installed 

- Ensure Python version is 3.11 

- Ensure OpenVINO is installed from the archive (OpenVINO 2024.1 for onnxruntime_openvino-1.18) 

- Build a wheel
      
    ```sh
    git clone https://github.com/microsoft/onnxruntime.git 
    ```
    ```sh
    cd onnxruntime && mkdir build 
    ```
    In Command Prompt initialize OpenVINO, for example using
    ```sh
    "C:\Program Files (x86)\Intel\openvino_2024\setupvars.bat"
    ``` 
    Build in Windows:
    ```sh
    .\build.bat --config Debug --use_openvino CPU --build_shared_lib --build_wheel --parallel --skip_tests
    ```
    Discover the wheel file in the path - e.g.: 
    ```sh
    \onnxruntime\build\Windows\Debug\Debug\dist\ 
    ```




