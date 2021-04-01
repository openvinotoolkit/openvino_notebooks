# 📚 OpenVINO Notebooks

> **NOTE**: The notebooks are currently in **beta**. We plan to publish a stable release this summer. You can submit [issues](https://github.com/openvinotoolkit/openvino_notebooks/issues) on GitHub and join our [developer Discord\*](https://discord.com/invite/pWGcWpyx7x) to get updates. We look forward to hearing from you!

A collection of ready-to-run Python\* notebooks for learning and experimenting with OpenVINO developer tools. The notebooks are meant to provide an introduction to OpenVINO basics and teach developers how to leverage our APIs for optimized deep learning inference in their applications.

## Getting Started

The notebooks are designed to run almost anywhere &mdash; on your laptop, a cloud VM, or a Docker container. Here's what you need to get started:

- CPU (64-bit)
- Windows\*, Linux\* or macOS\*
- Python\*

Before you proceed to the Installation Guide, please check the detailed System Requirements below.

## System Requirements

> **NOTE: Python 3.8 is not supported yet.** If you wish to run the notebooks on Ubuntu 20.04, please see our [guide](wiki_url) for installing Python 3.7 on Ubuntu 20.04.

The table below lists the supported operating systems and Python versions required to run the OpenVINO notebooks.

| Supported Operating System                                 | [Python\* Version (64-bit)](https://www.python.org/) |
| :--------------------------------------------------------- | :--------------------------------------------------- |
| Ubuntu\* 18.04 LTS, 64-bit                                 | 3.6, 3.7                                             |
| Ubuntu\* 20.04 LTS, 64-bit                                 | 3.6, 3.7                                             |
| Red Hat* Enterprise Linux* 8.2, 64-bit                     | 3.6, 3.7                                             |
| CentOS\* 7.4, 64-bit                                       | 3.6, 3.7                                             |
| macOS\* 10.15.x versions                                   | 3.6, 3.7                                             |
| Windows 10\*, 64-bit Pro, Enterprise or Education editions | 3.6, 3.7                                             |
| Windows Server\* 2016 or higher                            | 3.6, 3.7                                             |

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/openvinotoolkit/openvino_notebooks.git
```

### Step 2: Create a Virtual Environment

```bash
# NOTE: On Linux and macOS you may need to use python3 instead of python
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
# Always upgrade pip to the latest version
python -m pip install --upgrade pip
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
jupyter lab
```

In Jupyter Lab, select a notebook from the file browser using the left sidebar. Each notebook is located in a subdirectory within the `notebooks` directory.

## Troubleshooting

- On Ubuntu, if you see the error **"libpython3.7m.so.1.0: cannot open shared object file: No such object or directory"** please install the required package with `sudo apt install libpython3.7-dev`

- If you get an `ImportError`, doublecheck that you installed the kernel in Step 4. If necessary, choose the openvino*env kernel from the \_Kernel->Change Kernel* menu)

- On Linux and macOS you may need to type `python3` instead of `python` when creating your virtual environment

- You may also need to install [pip](https://pip.pypa.io/en/stable/installing/) and/or python-venv (depending on your Linux distribution)

- On Windows, if you have installed multiple versions of Python, use `py -3.7` when creating your virtual environment to specify a supported version (in this case 3.7)
