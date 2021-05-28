# 📚 OpenVINO Notebooks

A collection of ready-to-run Jupyter\* notebooks for learning and experimenting with the OpenVINO™ Toolkit. The notebooks provide an introduction to OpenVINO basics and teach developers how to leverage our API for optimized deep learning inference.

## 💻 Getting Started

The notebooks run almost anywhere &mdash; your laptop, a cloud VM, or even a Docker container. Here's what you need to get started:

- CPU (64-bit)
- Windows\*, Linux\* or macOS\*
- Python\* 3.6-3.8
- Git\*

Before you proceed to the [Installation Guide](#-installation-guide), please review the detailed [System Requirements](#%EF%B8%8F-system-requirements) below. The [Notebooks Wiki](https://github.com/openvinotoolkit/openvino_notebooks/wiki#guides-per-operating-system) has additional
details about required packages and links to the installers for Git and Python.
There is also a guide for running the notebooks in [Azure\* ML Studio](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML).

## ⚙️ System Requirements

> **NOTE:** Python 3.9 is not supported yet, but it will be very soon.

The table below lists the supported operating systems and Python versions required to run the OpenVINO notebooks.

| Supported Operating System                                 | [Python Version (64-bit)](https://www.python.org/) |
| :--------------------------------------------------------- | :------------------------------------------------- |
| Ubuntu\* 18.04 LTS, 64-bit                                 | 3.6, 3.7, 3.8                                      |
| Ubuntu\* 20.04 LTS, 64-bit                                 | 3.6, 3.7, 3.8                                      |
| Red Hat* Enterprise Linux* 8, 64-bit                       | 3.6, 3.8                                           |
| CentOS\* 7, 64-bit                                         | 3.6, 3.7, 3.8                                      |
| macOS\* 10.15.x versions                                   | 3.6, 3.7, 3.8                                      |
| Windows 10\*, 64-bit Pro, Enterprise or Education editions | 3.6, 3.7, 3.8\*\*                                  |
| Windows Server\* 2016 or higher                            | 3.6, 3.7, 3.8\*\*                                  |

> \*\*_At the moment, For Python 3.8 on Windows, OpenVINO requires installation of [Microsoft Visual C++ Redistributable](https://visualstudio.microsoft.com/downloads/#microsoft-visual-c-redistributable-for-visual-studio-2019). This is not required for Python 3.6 and 3.7, and will not be required for Python 3.8 in the next OpenVINO release._

## 📝 Installation Guides

| [Windows](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) | [Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu) | [macOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS) | [AzureML](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML) |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |

> **NOTE:** If OpenVINO is installed globally, please do not run any of these commands in a terminal where setupvars.bat or setupvars.sh are sourced.

Run each step below on a command-line interface. For Linux and macOS, use a Terminal. For Windows, we recommend using _Command Prompt (cmd.exe)_, not _PowerShell_.

### Step 1: Create a Virtual Environment

> **NOTE:** If you already installed openvino-dev and activated the openvino_env environment, you can skip to [Step 3](#step-3-clone-the-repository). If you use Anaconda, please see the [Conda guide](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Conda).

#### For Linux and macOS:

```bash
python3 -m venv openvino_env
```

#### For Windows:

```bash
python -m venv openvino_env
```

### Step 2: Activate the Environment

#### For Linux and macOS:

```bash
source openvino_env/bin/activate
```

#### For Windows:

```bash
openvino_env\Scripts\activate
```

### Step 3: Clone the Repository

```bash
git clone https://github.com/openvinotoolkit/openvino_notebooks.git
cd openvino_notebooks
```

### Step 4: Install the Packages

This step installs OpenVINO and dependencies like Jupyter Lab. First, upgrade pip to the latest version. Then, use pip's legacy dependency resolver to avoid dependency conflicts.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

### Step 5: Install the virtualenv Kernel in Jupyter

```bash
python -m ipykernel install --user --name openvino_env
```

### Step 6: Launch the Notebooks!

To launch a single notebook, like the Monodepth notebook

```bash
jupyter notebook notebooks/201-vision-monodepth/201-vision-monodepth.ipynb
```

To launch all notebooks in Jupyter Lab

```bash
jupyter lab notebooks
```

In Jupyter Lab, select a notebook from the file browser using the left sidebar. Each notebook is located in a subdirectory within the `notebooks` directory.

<img src="notebooks/jupyterlab.gif">

## 🧹 Cleaning Up

### Shut Down Jupyter Kernel

To end your Jupyter session, press `Ctrl-c`. This will prompt you to `Shutdown this Jupyter server (y/[n])?` enter `y` and hit `Enter`.

### Deactivate Virtual Environment

To deactivate your virtualenv, simply run `deactivate` from the terminal window where you activated `openvino_env`. This will deactivate your environment.

To reactivate your environment, simply repeat [Step 3](#step-3-activate-the-environment) from the Install Guide.
To start the notebooks again, type `jupyter lab` or `jupyter notebook` after activating the environment.

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

If these tips do not solve your problem, please open a [discussion topic](https://github.com/openvinotoolkit/openvino_notebooks/discussions)
or create an [issue](https://github.com/openvinotoolkit/openvino_notebooks/issues)!

- To check some common installation problems, run `python launch_notebooks.py`. This script is located in the openvino_notebooks directory.
  Please run it after activating the `openvino_env` virtual environment.
- If you get an `ImportError`, doublecheck that you installed the Jupyter kernel in [Step 5](#step-5-install-the-virtualenv-kernel-in-jupyter).
  If necessary, choose the openvino*env kernel from the \_Kernel->Change Kernel* menu) in Jupyter Lab or Jupyter Notebook

### Windows

- On Windows, if you have installed multiple versions of Python, use `py -3.7` when creating your virtual environment to specify a supported version (in this case 3.7).
- Please see the [wiki/Windows](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) page for information on how to install Python or Git.
- If you use Anaconda, you may need to add OpenVINO to your Windows PATH. See the [wiki/Conda](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Conda) page.
- If you see an error about needing to install C++, please either install
  [Microsoft Visual C++ Redistributable](https://visualstudio.microsoft.com/downloads/#microsoft-visual-c-redistributable-for-visual-studio-2019)
  or use Python 3.7, which does not have this requirement.

### Linux and macOS

- On Ubuntu, if you see the error **"libpython3.7m.so.1.0: cannot open shared object file: No such object or directory"** please install
  the required package using `apt install libpython3.7-dev`.
- See the [wiki/Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu) for all required Ubuntu packages.
- On Fedora*, Red Hat and Amazon* Linux you may need to install the OpenGL (Open Graphics Library) to use OpenCV. Please run `yum install mesa-libGL`
  before launching the notebooks.
- For macOS systems with Apple* M1, please see [community discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions/10) about using Rosetta* 2.

---

\* Other names and brands may be claimed as the property of others.
