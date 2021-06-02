# 📚 OpenVINO Notebooks

A collection of ready-to-run Jupyter\* notebooks for learning and experimenting with the OpenVINO™ Toolkit. The notebooks provide an introduction to OpenVINO basics and teach developers how to leverage our API for optimized deep learning inference.

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

OpenVINO Notebooks require Python and Git. See instructions below for your operating system:

| [Windows 10](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) | [Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu) | [macOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS) | [Red Hat](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [CentOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [Azure ML](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML) |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |

Once you have installed Python and Git, follow the steps below.

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
pip install -r requirements.txt --use-deprecated=legacy-resolver
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

<img src="https://user-images.githubusercontent.com/15709723/120404605-92c69600-c2fb-11eb-99be-5c6153da42b3.gif">

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

---

\* Other names and brands may be claimed as the property of others.
