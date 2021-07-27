[English](README.md) | 简体中文
 
# 📚 OpenVINO Notebooks

![GitHub release (latest by date)](https://img.shields.io/github/v/release/openvinotoolkit/openvino_notebooks)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/nbval.yml/badge.svg)
![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml/badge.svg)

一些可以运行的Jupyter\* notebooks，用于学习和试验OpenVINO™开发套件。这些notebooks旨在提供OpenVINO基础知识的介绍，并教开发人员如何利用我们的API在应用程序中优化深度学习推理。

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
