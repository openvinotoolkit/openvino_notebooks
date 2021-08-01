English | [简体中文](README_cn.md)

# 📚 OpenVINO™ Notebooks

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/nbval.yml/badge.svg)
![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml/badge.svg)

A collection of ready-to-run Jupyter\* notebooks for learning and experimenting with the OpenVINO™ Toolkit. The notebooks provide an introduction to OpenVINO basics and teach developers how to leverage our API for optimized deep learning inference.

## 📖 What's Inside

|                                                                                                                                                                Notebook                                                                                                                                                                 | Description                                                                                         |                                                                                                                         Preview                                                                                                                          |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                              [001-hello-world](001-hello-world/001-hello-world.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F001-hello-world%2F001-hello-world.ipynb)                                              | Classify an image with OpenVINO                                                                     |                                                               <img src="https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg" width=115>                                                                |
|                                           [002-openvino-api](002-openvino-api/002-openvino-api.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F002-openvino-api%2F002-openvino-api.ipynb)                                            | Learn the Inference Engine Python API                                                               |                                                                                                                                                                                                                                                          |
|                  [101-tensorflow-to-openvino](101-tensorflow-to-openvino/101-tensorflow-to-openvino.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F101-tensorflow-to-openvino%2F101-tensorflow-to-openvino.ipynb)                   | Convert a TensorFlow model to OpenVINO IR                                                           |                                                               <img src="https://user-images.githubusercontent.com/15709723/127779167-9d33dcc6-9001-4d74-a089-8248310092fe.png" width=250>                                                                |
|             [102-pytorch-onnx-to-openvino](102-pytorch-onnx-to-openvino/102-pytorch-onnx-to-openvino.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F102-pytorch-onnx-to-openvino%2F102-pytorch-onnx-to-openvino.ipynb)              | Convert a PyTorch model to OpenVINO IR                                                              |                                                                    <img src="https://user-images.githubusercontent.com/15709723/127779246-32e7392b-2d72-4a7d-b871-e79e7bfdd2e9.png" >                                                                    |
| [103-paddle-onnx-to-openvino](103-paddle-onnx-to-openvino/103-paddle-onnx-to-openvino-classification.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F103-paddle-onnx-to-openvino%2F103-paddle-onnx-to-openvino-classification.ipynb) | Convert a PaddlePaddle model to OpenVINO IR                                                         |                                                               <img src="https://user-images.githubusercontent.com/15709723/127779326-dc14653f-a960-4877-b529-86908a6f2a61.png" width=300>                                                                |
|                                              [104-model-tools](104-model-tools/104-model-tools.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F104-model-tools%2F104-model-tools.ipynb)                                              | Learn how to download, convert and benchmark models from the Open Model Zoo                         |                                                                                                                                                                                                                                                          |
|                                 [201-vision-monodepth](201-vision-monodepth/201-vision-monodepth.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F201-vision-monodepth%2F201-vision-monodepth.ipynb)                                  | Try out a monocular depth estimation model with images and video                                    |                                                               <img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif" width=200>                                                                |
|         [202-vision-superresolution-image](202-vision-superresolution/202-vision-superresolution-image.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-image.ipynb)          | Upscale a raw TIFF image using a super resolution model                                             |                                                  <img src="notebooks/202-vision-superresolution/data/tower.jpg" width="60">→<img src="notebooks/202-vision-superresolution/data/tower.jpg" width="120">                                                  |
|         [202-vision-superresolution-video](202-vision-superresolution/202-vision-superresolution-video.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-video.ipynb)          | Turn a 360p video into a 1080p using a super resolution model                                       | <img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width=60>→<img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width="120"> |
|           [205-vision-background-removal](205-vision-background-removal/205-vision-background-removal.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F205-vision-background-removal%2F205-vision-background-removal.ipynb)           | Remove and replace the background in an image with a salient object detection model                 |                                                               <img src="https://user-images.githubusercontent.com/15709723/125184237-f4b6cd00-e1d0-11eb-8e3b-d92c9a728372.png" width=250>                                                                |
|                  [206-vision-paddlegan-anime](206-vision-paddlegan-anime/206-vision-paddlegan-anime.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F206-vision-paddlegan-anime%2F206-vision-paddlegan-anime.ipynb)                   | Turn an image into anime using a generative adversarial network (GAN) model                         |                                                               <img src="https://user-images.githubusercontent.com/15709723/125184441-b4584e80-e1d2-11eb-8964-d8131cd97409.png" width=120>                                                                |
|                                                                                                               [301-tensorflow-training-openvino](301-tensorflow-training-openvino/301-tensorflow-training-openvino.ipynb)                                                                                                               | Train a flower classification model from TensorFlow using an Intel CPU, then convert to OpenVINO IR |                                                               <img src="https://user-images.githubusercontent.com/15709723/127779607-8fa34947-1c35-4260-8d04-981c41a2a2cc.png" width=300>                                                                |
|                                                                                                           [301-tensorflow-training-openvino-pot](301-tensorflow-training-openvino/301-tensorflow-training-openvino-pot.ipynb)                                                                                                           | Use OpenVINO's Post-training Optimization Tool (POT) to quantize the flowers model                  |                                                                      <img src="https://upload.wikimedia.org/wikipedia/commons/4/48/A_Close_Up_Photo_of_a_Dandelion.jpg" width=140>                                                                       |

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

OpenVINO Notebooks require Python and Git. For Python 3.8, C++ is also required. See instructions below for your operating system:

| [Windows 10](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) | [Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu) | [macOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS) | [Red Hat](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [CentOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [Azure ML](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML) |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |

Once you have installed Python, Git and C++, follow the steps below.

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
