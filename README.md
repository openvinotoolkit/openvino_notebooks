English | [简体中文](README_cn.md)

# 📚 OpenVINO™ Notebooks

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/nbval.yml/badge.svg)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/nbval.yml?query=branch%3Amain+event%3Apush)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml/badge.svg)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/nbval.yml?query=branch%3Amain+event%3Apush)

A collection of ready-to-run Jupyter\* notebooks for learning and experimenting with the OpenVINO™ Toolkit. The notebooks provide an introduction to OpenVINO basics and teach developers how to leverage our API for optimized deep learning inference.

## 📖 What's Inside

Notebooks with a ![binder logo](https://mybinder.org/badge_logo.svg) button can be run without installing anything. [Binder](https://mybinder.org/) is a free online service with limited resources. For the best performance, please follow the [Installation Guide](#-installation-guide) and run the notebooks locally.

### Getting Started

Brief tutorials that demonstrate how to use OpenVINO's Python API for inference.
| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [001-hello-world](notebooks/001-hello-world/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F001-hello-world%2F001-hello-world.ipynb) | Classify an image with OpenVINO | <img src="https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg" width=140> |
| [002-openvino-api](notebooks/002-openvino-api/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F002-openvino-api%2F002-openvino-api.ipynb) | Learn the OpenVINO Python API | <img src="https://user-images.githubusercontent.com/15709723/127787560-d8ec4d92-b4a0-411f-84aa-007e90faba98.png" width=250> |
| [003-hello-segmentation](notebooks/003-hello-segmentation/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F003-hello-segmentation%2F003-hello-segmentation.ipynb) | Semantic segmentation with OpenVINO | <img src="https://user-images.githubusercontent.com/15709723/128290691-e2eb875c-775e-4f4d-a2f4-15134044b4bb.png" width=150> |
| [004-hello-detection](notebooks/004-hello-detection/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F004-hello-detection%2F004-hello-detection.ipynb) | Text detection with OpenVINO | <img src="https://user-images.githubusercontent.com/36741649/128489933-bf215a3f-06fa-4918-8833-cb0bf9fb1cc7.jpg" width=150> |

### Convert & Optimize

Tutorials that explain how to optimize and quantize models with OpenVINO tools.
| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [101-tensorflow-to-openvino](notebooks/101-tensorflow-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F101-tensorflow-to-openvino%2F101-tensorflow-to-openvino.ipynb) | Convert TensorFlow models to OpenVINO IR | <img src="https://user-images.githubusercontent.com/15709723/127779167-9d33dcc6-9001-4d74-a089-8248310092fe.png" width=250> |
| [102-pytorch-onnx-to-openvino](notebooks/102-pytorch-onnx-to-openvino/) | Convert PyTorch models to OpenVINO IR | <img src="https://user-images.githubusercontent.com/15709723/127779246-32e7392b-2d72-4a7d-b871-e79e7bfdd2e9.png" width=300 > |
| [103-paddle-onnx-to-openvino](notebooks/103-paddle-onnx-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F103-paddle-onnx-to-openvino%2F103-paddle-onnx-to-openvino-classification.ipynb) | Convert PaddlePaddle models to OpenVINO IR | <img src="https://user-images.githubusercontent.com/15709723/127779326-dc14653f-a960-4877-b529-86908a6f2a61.png" width=300> |
| [104-model-tools](notebooks/104-model-tools/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F104-model-tools%2F104-model-tools.ipynb) | Download, convert and benchmark models from Open Model Zoo | |
| [105-language-quantize-bert](notebooks/105-language-quantize-bert/) | Optimize and quantize a pre-trained BERT model ||
| [110-ct-segmentation-quantize](notebooks/110-ct-segmentation-quantize/110-ct-segmentation-quantize.ipynb)<br> | Quantize a kidney segmentation model and show live inference | |


### Model Demos

Demos that demonstrate inference on a particular model.
| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [201-vision-monodepth](notebooks/201-vision-monodepth/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F201-vision-monodepth%2F201-vision-monodepth.ipynb) | Monocular depth estimation with images and video | <img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif" width=250> |
| [202-vision-superresolution-image](notebooks/202-vision-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-image.ipynb) | Upscale raw images with a super resolution model | <img src="notebooks/202-vision-superresolution/data/tower.jpg" width="70">→<img src="notebooks/202-vision-superresolution/data/tower.jpg" width="130"> |
| [202-vision-superresolution-video](notebooks/202-vision-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-video.ipynb) | Turn 360p into 1080p video using a super resolution model | <img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width=80>→<img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width="125"> |
| [205-vision-background-removal](notebooks/205-vision-background-removal/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F205-vision-background-removal%2F205-vision-background-removal.ipynb) | Remove and replace the background in an image using salient object detection | <img src="https://user-images.githubusercontent.com/15709723/125184237-f4b6cd00-e1d0-11eb-8e3b-d92c9a728372.png" width=455> |
| [206-vision-paddlegan-anime](notebooks/206-vision-paddlegan-anime/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F206-vision-paddlegan-anime%2F206-vision-paddlegan-anime.ipynb) | Turn an image into anime using a GAN | <img src="https://user-images.githubusercontent.com/15709723/127788059-1f069ae1-8705-4972-b50e-6314a6f36632.jpeg" width=100>→<img src="https://user-images.githubusercontent.com/15709723/125184441-b4584e80-e1d2-11eb-8964-d8131cd97409.png" width=100> |
| [207-vision-paddlegan-superresolution](notebooks/207-vision-paddlegan-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F207-vision-paddlegan-superresolution%2F207-vision-paddlegan-superresolution.ipynb)| Upscale small images with superresolution using a PaddleGAN model| |
| [208-optical-character-recognition](notebooks/208-optical-character-recognition/)<br> | Annotate text on images using text recognition resnet | <img src="https://user-images.githubusercontent.com/36741649/129315292-a37266dc-dfb2-4749-bca5-2ac9c1e93d64.jpg" width=225> |
| [210-ct-scan-live-inference](notebooks/210-ct-scan-live-inference/210-ct-scan-live-inference.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F210-ct-scan-live-inference%2F210-ct-scan-live-inference.ipynb)| Show live inference on segmentation of CT-scan data |<img src="https://user-images.githubusercontent.com/15709723/134784204-cf8f7800-b84c-47f5-a1d8-25a9afab88f8.gif" width=225>|
### Model Training

Tutorials that include code to train neural networks.
| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [301-tensorflow-training-openvino](notebooks/301-tensorflow-training-openvino/) | Train a flower classification model from TensorFlow, then convert to OpenVINO IR | <img src="https://user-images.githubusercontent.com/15709723/127779607-8fa34947-1c35-4260-8d04-981c41a2a2cc.png" width=390> |
| [301-tensorflow-training-openvino-pot](notebooks/301-tensorflow-training-openvino/) | Use Post-training Optimization Tool (POT) to quantize the flowers model | |
| [302-pytorch-quantization-aware-training](notebooks/302-pytorch-quantization-aware-training) | Use Neural Network Compression Framework (NNCF) to quantize PyTorch model | |
| [305-tensorflow-quantization-aware-training](notebooks/305-tensorflow-quantization-aware-training) | Use Neural Network Compression Framework (NNCF) to quantize TensorFlow model | |

### Live Demos

Live inference demos that require a webcam.
| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [401-object-detection-webcam](notebooks/401-object-detection-webcam/) | Object detection with a webcam or video file | <img src="https://user-images.githubusercontent.com/4547501/137755014-6f174efb-f378-4fe2-9edd-cab7aab0df5f.jpg" width=225> |
| [402-pose-etimation-webcam](notebooks/402-pose-estimation-webcam/) | Human pose estimation with a webcam or video file | <img src="https://user-images.githubusercontent.com/4547501/138267961-41d754e7-59db-49f6-b700-63c3a636fad7.gif" width=225> |


## ⚙️ System Requirements

The notebooks run almost anywhere &mdash; your laptop, a cloud VM, or even a Docker container. The table below lists the supported operating systems and Python versions. **Note:** Python 3.9 is not supported yet, but it will be soon.

| Supported Operating System                                 | [Python Version (64-bit)](https://www.python.org/) |
| :--------------------------------------------------------- | :------------------------------------------------- |
| Ubuntu\* 18.04 LTS, 64-bit                                 | 3.6, 3.7, 3.8                                      |
| Ubuntu\* 20.04 LTS, 64-bit                                 | 3.6, 3.7, 3.8                                      |
| Red Hat* Enterprise Linux* 8, 64-bit                       | 3.6, 3.8                                           |
| CentOS\* 7, 64-bit                                         | 3.6, 3.7, 3.8                                      |
| macOS\* 10.15.x versions                                   | 3.6, 3.7, 3.8                                      |
| Windows 10\*, 64-bit Pro, Enterprise or Education editions | 3.6, 3.7, 3.8                                      |
| Windows Server\* 2016 or higher                            | 3.6, 3.7, 3.8                                      |

## 📝 Installation Guide

OpenVINO Notebooks require Python and Git. For Python 3.8, C++ is also required. Select a guide for your operating system or environment:

| [Windows 10](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) | [Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu) | [macOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS) | [Red Hat](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [CentOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [Azure ML](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML) | [Docker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Docker) |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |

Or, if you have already installed Python, Git and C++, please follow the steps below.

### Step 1: Create and Activate `openvino_env` Environment

#### Linux and macOS Commands:

```bash
python3 -m venv openvino_env
source openvino_env/bin/activate
```

#### Windows Commands:

```bash
python -m venv openvino_env
openvino_env\Scripts\activate
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/openvinotoolkit/openvino_notebooks.git
cd openvino_notebooks
```

### Step 3: Install and Launch the Notebooks

Upgrade pip to the latest version. Use pip's legacy dependency resolver to avoid dependency conflicts

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name openvino_env
```

## 💻 Run the Notebooks

### To Launch a Single Notebook

If you wish to launch only one notebook, like the Monodepth notebook, run the command below.

```bash
jupyter notebook notebooks/201-vision-monodepth/201-vision-monodepth.ipynb
```

### To Launch all Notebooks

```bash
jupyter lab notebooks
```

In your browser, select a notebook from the file browser in Jupyter Lab using the left sidebar. Each tutorial is located in a subdirectory within the `notebooks` directory.

<img src="https://user-images.githubusercontent.com/15709723/120527271-006fd200-c38f-11eb-9935-2d36d50bab9f.gif">

## 🧹 Cleaning Up

### Shut Down Jupyter Kernel

To end your Jupyter session, press `Ctrl-c`. This will prompt you to `Shutdown this Jupyter server (y/[n])?` enter `y` and hit `Enter`.

### Deactivate Virtual Environment

To deactivate your virtualenv, simply run `deactivate` from the terminal window where you activated `openvino_env`. This will deactivate your environment.

To reactivate your environment, run `source openvino_env/bin/activate` on Linux or `openvino_env\Scripts\activate` on Windows, then type `jupyter lab` or `jupyter notebook` to launch the notebooks again.

### Delete Virtual Environment _(Optional)_

To remove your virtual environment, simply delete the `openvino_env` directory:

#### On Linux and macOS:

```bash
rm -rf openvino_env
```

#### On Windows:

```bash
rmdir /s openvino_env
```

### Remove openvino_env Kernel from Jupyter

```bash
jupyter kernelspec remove openvino_env
```

## ⚠️ Troubleshooting

If these tips do not solve your problem, please open a [discussion topic](https://github.com/openvinotoolkit/openvino_notebooks/discussions)
or create an [issue](https://github.com/openvinotoolkit/openvino_notebooks/issues)!

- To check some common installation problems, run `python check_install.py`. This script is located in the openvino_notebooks directory.
  Please run it after activating the `openvino_env` virtual environment.
- If you get an `ImportError`, doublecheck that you installed the Jupyter kernel. If necessary, choose the openvino*env kernel from the \_Kernel->Change Kernel* menu) in Jupyter Lab or Jupyter Notebook
- If OpenVINO is installed globally, do not run installation commands in a terminal where setupvars.bat or setupvars.sh are sourced.
- For Windows installation, we recommend using _Command Prompt (cmd.exe)_, not _PowerShell_.

---

\* Other names and brands may be claimed as the property of others.
