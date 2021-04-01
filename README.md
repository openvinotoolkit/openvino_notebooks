# OpenVINO Notebooks

> **NOTE**: The notebooks are currently in **beta**. We plan to publish a stable release this summer. In the meantime, please try the beta notebooks and provide feedback. You can submit [issues](https://github.com/openvinotoolkit/openvino_notebooks/issues) on GitHub and join our [developer Discord\*](https://discord.com/invite/pWGcWpyx7x) to get updates. We look forward to hearing from you!

A collection of ready-to-run Python\* notebooks for learning and experimenting with the OpenVINO Developer Tools. The notebooks are meant to provide an introduction to OpenVINO basics and teach developers how to leverage our APIs for optimized deep learning inference in your application.

## Getting Started

The notebooks are designed to run almost anywhere, on your laptop, PC, in the cloud, on a VM, bare-metal or inside a container. Here's what you need before getting started:

- CPU (64-bit)
- Windows\*, Linux\* or macOS\*
- Python\*

Before you proceed to the Installation Guide, check the detailed System Requirements below.

## System Requirements

> **NOTE**: Python 3.8 on Linux operating systems is not supported yet. If you wish to run the notebooks on Ubuntu 20.04, please see our [guide](wiki_url) for installing Python 3.7 on Ubuntu 20.04.

The table below lists the supported operating systems and Python\* versions required to run the OpenVINO notebooks.

| Supported Operating System                                                                                  | [Python\* Version (64-bit)](https://www.python.org/) |
| :---------------------------------------------------------------------------------------------------------- | :--------------------------------------------------- |
| Ubuntu\* 18.04 long-term support (LTS), 64-bit                                                              | 3.6, 3.7                                             |
| Ubuntu\* 20.04 long-term support (LTS), 64-bit                                                              | 3.6, 3.7                                             |
| Red Hat* Enterprise Linux* 8.2, 64-bit                                                                      | 3.6, 3.7                                             |
| CentOS\* 7.4, 64-bit                                                                                        | 3.6, 3.7                                             |
| macOS\* 10.15.x versions                                                                                    | 3.6, 3.7, 3.8                                        |
| Windows 10\*, 64-bit Pro, Enterprise or Education (1607 Anniversary Update, Build 14393 or higher) editions | 3.6, 3.7, 3.8                                        |
| Windows Server\* 2016 or higher                                                                             | 3.6, 3.7, 3.8                                        |

## Installation Guide

### Step 1: Create a Virtual Environment

> **NOTE FOR WINDOWS:** if you have installed multiple versions of Python, use `py -3.7` to specify a supported version (in this case 3.7)

> **NOTE FOR LINUX/MAC:** you may need to type `python3` instead of `python` and you may also need to install [pip](https://pip.pypa.io/en/stable/installing/) and/or python-venv (depending on your Linux distribution).

```
python -m venv openvino_env
```

### Step 2: Activate the Environment

#### For Linux and macOS:

```
source openvino_env/bin/activate
```

#### For Windows:

```
openvino_env\Scripts\activate
```

### Step 3: Install the Packages

> **NOTE:** Please install pip with this specific version to ensure compatibility with OpenVINO versions and all dependencies.

```
python -m pip install --upgrade pip==20.1.1
pip install jupyterlab openvino-dev
```

### Step 4: Install the virtualenv Kernel in Jupyter

```
python -m ipykernel install --user --name openvino_env
```

## 5. Start the Notebook(s)!

#### To launch a single notebook:

```
jupyter notebook <notebook_filename>
```

#### or to launch Jupyter Lab IDE:

```
jupyter lab
```

In Jupyter Lab, select a notebook from the file browser using the left sidebar.

## Troubleshooting

On Linux, if you get the error "libpython3.7m.so.1.0: cannot open shared object file: No such object or directory" install the required package with `sudo apt install libpython3.7-dev`

If you get an `ImportError`, doublecheck that you installed the kernel in step 4. If necessary, choose the openvino*env kernel from the \_Kernel->Change Kernel* menu)
