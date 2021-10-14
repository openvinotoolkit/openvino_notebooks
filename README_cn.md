[English](README.md) | 简体中文
 
# 📚 OpenVINO Notebooks

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/nbval.yml/badge.svg)
![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml/badge.svg)

一些可以运行的Jupyter\* notebooks，用于学习和试验OpenVINO™开发套件。这些notebooks旨在提供OpenVINO基础知识的介绍，并教开发人员如何利用我们的API在应用程序中优化深度学习推理。

### 让我们开始吧

这个简短的教程将指导我们如果通过Openvino的Python API进行推理
| Notebook | 说明 | 预览 |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [001-hello-world](notebooks/001-hello-world/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F001-hello-world%2F001-hello-world.ipynb) | 14行代码实现视觉分类检测应用 | <img src="https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg" width=140> |
| [002-openvino-api](notebooks/002-openvino-api/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F002-openvino-api%2F002-openvino-api.ipynb) | Openvino python api介绍 | <img src="https://user-images.githubusercontent.com/15709723/127787560-d8ec4d92-b4a0-411f-84aa-007e90faba98.png" width=250> |
| [003-hello-segmentation](notebooks/003-hello-segmentation/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F003-hello-segmentation%2F003-hello-segmentation.ipynb) | 基于Openvino的视觉语义分割应用 | <img src="https://user-images.githubusercontent.com/15709723/128290691-e2eb875c-775e-4f4d-a2f4-15134044b4bb.png" width=150> |
| [004-hello-detection](notebooks/004-hello-detection/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F004-hello-detection%2F004-hello-detection.ipynb) | 基于Openvino的文字识别应用 | <img src="https://user-images.githubusercontent.com/36741649/128489933-bf215a3f-06fa-4918-8833-cb0bf9fb1cc7.jpg" width=150> |

### Convert & Optimize

这个教程将说明如何利用Openvino工具来量化和优化一个深度学习模型
| Notebook | 说明 | 预览 |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [101-tensorflow-to-openvino](notebooks/101-tensorflow-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F101-tensorflow-to-openvino%2F101-tensorflow-to-openvino.ipynb) | 基于Tensorflow预训练模型，实现分类检测部署 | <img src="https://user-images.githubusercontent.com/15709723/127779167-9d33dcc6-9001-4d74-a089-8248310092fe.png" width=250> |
| [102-pytorch-onnx-to-openvino](notebooks/102-pytorch-onnx-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F102-pytorch-onnx-to-openvino%2F102-pytorch-onnx-to-openvino.ipynb) | 基于Pytorch预训练模型，实现语义分割部署 | <img src="https://user-images.githubusercontent.com/15709723/127779246-32e7392b-2d72-4a7d-b871-e79e7bfdd2e9.png" width=300 > |
| [103-paddle-onnx-to-openvino](notebooks/103-paddle-onnx-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F103-paddle-onnx-to-openvino%2F103-paddle-onnx-to-openvino-classification.ipynb) | 基于PadlePadle预训练模型，实现分类检测部署 | <img src="https://user-images.githubusercontent.com/15709723/127779326-dc14653f-a960-4877-b529-86908a6f2a61.png" width=300> |
| [104-model-tools](notebooks/104-model-tools/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F104-model-tools%2F104-model-tools.ipynb) | Openvino模型的下载与评估 | |
| [105-language-quantize-bert](notebooks/105-language-quantize-bert/) | BERT预训练模型的优化与量化 ||

### Model Demos

特定模型的推理示例
| Notebook | 说明 | 预览 |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [201-vision-monodepth](notebooks/201-vision-monodepth/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F201-vision-monodepth%2F201-vision-monodepth.ipynb) | 单目深度检测应用实现 | <img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif" width=250> |
| [202-vision-superresolution-image](notebooks/202-vision-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-image.ipynb) | 图像超分辨率应用实现 | <img src="notebooks/202-vision-superresolution/data/tower.jpg" width="70">→<img src="notebooks/202-vision-superresolution/data/tower.jpg" width="130"> |
| [202-vision-superresolution-video](notebooks/202-vision-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-video.ipynb) | 视频超分辨率应用实现 | <img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width=80>→<img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width="125"> |
| [205-vision-background-removal](notebooks/205-vision-background-removal/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F205-vision-background-removal%2F205-vision-background-removal.ipynb) | 图像背景替换的应用实现 | <img src="https://user-images.githubusercontent.com/15709723/125184237-f4b6cd00-e1d0-11eb-8e3b-d92c9a728372.png" width=455> |
| [206-vision-paddlegan-anime](notebooks/206-vision-paddlegan-anime/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F206-vision-paddlegan-anime%2F206-vision-paddlegan-anime.ipynb) | 基于GAN的图片风格转换的应用实现 | <img src="https://user-images.githubusercontent.com/15709723/127788059-1f069ae1-8705-4972-b50e-6314a6f36632.jpeg" width=100>→<img src="https://user-images.githubusercontent.com/15709723/125184441-b4584e80-e1d2-11eb-8964-d8131cd97409.png" width=100> |
| [207-vision-paddlegan-superresolution](notebooks/207-vision-paddlegan-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F207-vision-paddlegan-superresolution%2F207-vision-paddlegan-superresolution.ipynb)| 基于GAN的图像超分辨率应用实现 | |
| [208-optical-character-recognition](notebooks/208-optical-character-recognition/)<br> | 文字识别应用实现 | <img src="https://user-images.githubusercontent.com/36741649/129315292-a37266dc-dfb2-4749-bca5-2ac9c1e93d64.jpg" width=225> |

### Model Training

这个教程将说明如何训练一个网络
| Notebook | 说明 | 预览 |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [301-tensorflow-training-openvino](notebooks/301-tensorflow-training-openvino/) | 基于Tensorflow 的模型训练及优化部署 | <img src="https://user-images.githubusercontent.com/15709723/127779607-8fa34947-1c35-4260-8d04-981c41a2a2cc.png" width=390> |
| [301-tensorflow-training-openvino-pot](notebooks/301-tensorflow-training-openvino/) | 基于POT工具的模型量化 | |
| [302-pytorch-quantization-aware-training](notebooks/302-pytorch-quantization-aware-training) | 基于NNCF工具的模型压缩 | |

### Live Demos

基于网络摄像头的实时推理示例
| Notebook | 说明 | 预览 |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [402-pose-etimation-webcam](notebooks/402-pose-estimation-webcam/) | 基于openvino人体姿态评估 | <img src="https://user-images.githubusercontent.com/4547501/134550328-a5c99d22-ae60-4281-8120-a8f06a17b96a.png" width=225> |

## ⚙️ 系统需求

这些notebooks几乎可以在任何地方运行—你的笔记本电脑，一个云虚拟机，甚至一个Docker容器。下表是目前支持的操作系统及Python版本。**注**：Python3.9目前还不支持，不过即将支持。

| Supported Operating System                                 | [Python Version (64-bit)](https://www.python.org/) |
| :--------------------------------------------------------- | :------------------------------------------------- |
| Ubuntu\* 18.04 LTS, 64-bit                                 | 3.6, 3.7, 3.8                                      |
| Ubuntu\* 20.04 LTS, 64-bit                                 | 3.6, 3.7, 3.8                                      |
| Red Hat* Enterprise Linux* 8, 64-bit                       | 3.6, 3.8                                           |
| CentOS\* 7, 64-bit                                         | 3.6, 3.7, 3.8                                      |
| macOS\* 10.15.x versions                                   | 3.6, 3.7, 3.8                                      |
| Windows 10\*, 64-bit Pro, Enterprise or Education editions | 3.6, 3.7, 3.8                                      |
| Windows Server\* 2016 or higher                            | 3.6, 3.7, 3.8                                      |

## 📝 安装指南

运行OpenVINO Notebooks需要预装Python和Git， 针对不同操作系统的安装参考以下英语指南：

| [Windows 10](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) | [Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu) | [macOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS) | [Red Hat](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [CentOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [Azure ML](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML) |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |

Python和Git安装完成后，参考以下步骤：

### Step 1: 创建并激活 `openvino_env` 虚拟环境

#### Linux 和 macOS 命令:

```bash
python3 -m venv openvino_env
source openvino_env/bin/activate
```

#### Windows 命令:

```bash
python -m venv openvino_env
openvino_env\Scripts\activate
```

### Step 2: 获取源码

```bash
git clone https://github.com/openvinotoolkit/openvino_notebooks.git
cd openvino_notebooks
```

### Step 3: 安装并启动 Notebooks

将pip升级到最新版本。

```bash
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m ipykernel install --user --name openvino_env
```

## 💻 运行 Notebooks

### 启动单个 Notebook

如果你希望启动单个的notebook（如：Monodepth notebook），运行以下命令： 

```bash
jupyter notebook notebooks/201-vision-monodepth/201-vision-monodepth.ipynb
```

### 启动所有 Notebooks

```bash
jupyter lab notebooks
```

在浏览器中，从Jupyter Lab侧边栏的文件浏览器中选择一个notebook文件，每个notebook文件都位于`notebooks`目录中的子目录中。

<img src="https://user-images.githubusercontent.com/15709723/120527271-006fd200-c38f-11eb-9935-2d36d50bab9f.gif">

## 🧹 清理

### 停止 Jupyter Kernel

按 `Ctrl-c` 结束 Jupyter session，会弹出一个提示框 `Shutdown this Jupyter server (y/[n])?` 输入 `y` 并按 `回车`。

### 注销虚拟环境

注销该虚拟环境：只需在激活了 `openvino_env` 的终端窗口中运行 `deactivate` 即可。

重新激活环境：在Linux上运行 `source openvino_env/bin/activate` 或者在Windows上运行 `openvino_env\Scripts\activate` 即可，然后输入 `jupyter lab` 或 `jupyter notebook` 即可重新运行notebooks。

### 删除虚拟环境_（可选）_

直接删除 `openvino_env` 目录即可删除虚拟环境：

#### Linux 和 macOS:

```bash
rm -rf openvino_env
```

#### Windows:

```bash
rmdir /s openvino_env
```

### 从Jupyter中移除openvino_env Kernel

```bash
jupyter kernelspec remove openvino_env
```

## ⚠️ 故障排除

如果以下方法无法解决您的问题，欢迎创建一个 [讨论话题](https://github.com/openvinotoolkit/openvino_notebooks/discussions)  或  [issue](https://github.com/openvinotoolkit/openvino_notebooks/issues) !

- 运行 `python check_install.py` 可以帮助检查一些常见的安装问题，该脚本位于openvino_notebooks 目录中。

  记得运行该脚本之前先激活 `openvino_env` 虚拟环境。

- 如果出现 `ImportError` ，请检查是否安装了 Jupyter Kernel。如需手动设置kernel，从 Jupyter Lab 或 Jupyter Notebook 的_Kernel->Change Kernel_菜单中选择openvino_env内核。

- 如果OpenVINO是全局安装的，不要在执行了setupvars.bat或setupvars.sh的终端中运行安装命令。

- 对于Windows系统，我们建议使用_Command Prompt (cmd.exe)_，而不是_PowerShell_。

---

\* Other names and brands may be claimed as the property of others.
