# 📚 OpenVINO Notebooks

> 🚧 Notebooks are currently in **beta**. We plan to publish a stable release this summer. Please submit [issues](https://github.com/openvinotoolkit/openvino_notebooks/issues) on GitHub, start a [discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions) or join our [Unofficial Developer Discord Server\*](https://discord.gg/xJK7SEC54c) to stay in touch.

A collection of ready-to-run Python\* notebooks for learning and experimenting with OpenVINO developer tools. The notebooks are meant to provide an introduction to OpenVINO basics and teach developers how to leverage our APIs for optimized deep learning inference in their applications.

## 💻 Getting Started

The notebooks are designed to run almost anywhere &mdash; your laptop, a cloud VM, or even a Docker container. Here's what you need to get started:

- CPU (64-bit)
- Windows\*, Linux\* or macOS\*
- Python\* 3.6-3.8

Before you proceed to the [Installation Guide](#-installation-guide), please review the detailed [System Requirements](#%EF%B8%8F-system-requirements) below.

## ⚙️ System Requirements

The table below lists the supported operating systems and Python versions required to run the OpenVINO notebooks.

| Supported Operating System                                 | [Python\* Version (64-bit)](https://www.python.org/) |
| :--------------------------------------------------------- | :--------------------------------------------------- |
| Ubuntu\* 18.04 LTS, 64-bit                                 | 3.6, 3.7, 3.8                                        |
| Ubuntu\* 20.04 LTS, 64-bit                                 | 3.6, 3.7, 3.8                                        |
| Red Hat* Enterprise Linux* 8, 64-bit                       | 3.6, 3.8                                             |
| CentOS\* 7, 64-bit                                         | 3.6, 3.7, 3.8                                        |
| macOS\* 10.15.x versions                                   | 3.6, 3.7, 3.8                                        |
| Windows 10\*, 64-bit Pro, Enterprise or Education editions | 3.6, 3.7, 3.8                                        |
| Windows Server\* 2016 or higher                            | 3.6, 3.7, 3.8                                        |

## 📝 Installation Guide

> **NOTE:** If OpenVINO is installed globally, please do not run any of these commands in a terminal where setupvars.bat or setupvars.sh are sourced. For Windows, we recommend using _Command Prompt (cmd.exe)_, not _PowerShell_.

### Step 1: Clone the Repository

```bash
git clone https://github.com/openvinotoolkit/openvino_notebooks.git
```

### Step 2: Create a Virtual Environment

```bash
# Linux and macOS may require typing python3 instead of python
cd openvino_notebooks
python -m venv openvino_env
```

### Step 3: Activate the Environment

#### For Linux and macOS:

```bash
source openvino_env/bin/activate
```

#### For Windows:

```bash
openvino_env\Scripts\activate
```

### Step 4: Install the Packages

#### Installs OpenVINO tools and dependencies like Jupyter Lab:

```bash
# Upgrade pip to the 20.2.* version to avoid dependency issues
python -m pip install --upgrade pip==20.2.4
pip install -r requirements.txt
```

### Step 5: Install the virtualenv Kernel in Jupyter

```bash
python -m ipykernel install --user --name openvino_env
```

### Step 6: Launch the Notebooks!

```bash
# To launch a single notebook
jupyter notebook <notebook_filename>

# To launch all notebooks in Jupyter Lab
jupyter lab notebooks
```

In Jupyter Lab, select a notebook from the file browser using the left sidebar. Each notebook is located in a subdirectory within the `notebooks` directory.

<img src="notebooks/jupyterlab.gif">

## 🧹 Cleaning Up

### Shut Down Jupyter Kernel

To end your Jupyter session, press `Ctrl-c`. This will prompt you to `Shutdown this Jupyter server (y/[n])?` enter `y` and hit `Enter`.

### Deativate Virtual Environment

To deactivate your virtualenv, simply run `deactivate` from the terminal window where you activated `openvino_env`. This will deactivate your environment.

To reactivate your environment, simply repeat [Step 3](#step-3-activate-the-environment) from the Install Guide.

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

### Remove `openvino_env` Kernel from Jupyter

```bash
jupyter kernelspec remove openvino_env
```

## ⚠️ Troubleshooting

- On Ubuntu, if you see the error **"libpython3.7m.so.1.0: cannot open shared object file: No such object or directory"** please install the required package using `apt install libpython3.7-dev`

- If you get an `ImportError`, doublecheck that you installed the kernel in [Step 5](#step-5-install-the-virtualenv-kernel-in-jupyter). If necessary, choose the openvino*env kernel from the \_Kernel->Change Kernel* menu)

- On Linux and macOS you may need to type `python3` instead of `python` when creating your virtual environment

- On Linux and macOS you may need to install [pip](https://pip.pypa.io/en/stable/installing/) and/or python-venv (depending on your Linux distribution)

- On Windows, if you have installed multiple versions of Python, use `py -3.7` when creating your virtual environment to specify a supported version (in this case 3.7)

- On Fedora*, Red Hat and Amazon* Linux you may need to install the OpenGL (Open Graphics Library) to use OpenCV. Please run `yum install mesa-libGL` before launching the notebooks.

- For macOS systems with Apple* M1, please see [community discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions/10) about using Rosetta* 2.

---

\* Other names and brands may be claimed as the property of others.
