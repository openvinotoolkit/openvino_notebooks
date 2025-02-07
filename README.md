English | [简体中文](README_cn.md)

<h1>📚 OpenVINO™ Notebooks</h1>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/LICENSE)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/treon_precommit.yml/badge.svg?event=push)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/treon_precommit.yml?query=event%3Apush)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml/badge.svg?event=push)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml?query=event%3Apush)

A collection of ready-to-run Jupyter notebooks for learning and experimenting with the OpenVINO™ Toolkit. The notebooks provide an introduction to OpenVINO basics and teach developers how to leverage our API for optimized deep learning inference.

🚀 Checkout interactive GitHub pages application for navigation between OpenVINO™ Notebooks content:
[OpenVINO™ Notebooks at GitHub Pages](https://openvinotoolkit.github.io/openvino_notebooks/)

[![notebooks-selector-preview](https://github.com/openvinotoolkit/openvino_notebooks/assets/41733560/a69efb78-1637-404c-b5ef-63974db2bf1b)](https://openvinotoolkit.github.io/openvino_notebooks/)

List of all notebooks is available in [index file](./notebooks/README.md).

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

## Table of Contents

- [Table of Contents](#table-of-contents)
- [📝 Installation Guide](#-installation-guide)
- [🚀 Getting Started](#-getting-started)
- [⚙️ System Requirements](#️-system-requirements)
- [💻 Run the Notebooks](#-run-the-notebooks)
	- [To Launch a Single Notebook](#to-launch-a-single-notebook)
	- [To Launch all Notebooks](#to-launch-all-notebooks)
- [🧹 Cleaning Up](#-cleaning-up)
- [⚠️ Troubleshooting](#️-troubleshooting)
- [📊 Telemetry](#-telemetry)
- [📚 Additional Resources](#-additional-resources)
- [🧑‍💻 Contributors](#-contributors)
- [❓ FAQ](#-faq)

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

<div id='-installation-guide'/>

## 📝 Installation Guide

OpenVINO Notebooks require Python and Git. To get started, select the guide for your operating system or environment:

| [Windows](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) | [Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu) | [macOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS) | [Red Hat](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [CentOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [Azure ML](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML) | [Docker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Docker) | [Amazon SageMaker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/SageMaker) |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

<div id='-getting-started'/>

## 🚀 Getting Started

Explore Jupyter notebooks using this [page](https://openvinotoolkit.github.io/openvino_notebooks/), select one related to your needs or give them all a try. Good Luck!

**NOTE: The main branch of this repository was updated to support the new OpenVINO 2025.0 release.** To upgrade to the new release version, please run `pip install --upgrade -r requirements.txt` in your `openvino_env` virtual environment. If you need to install for the first time, see the [Installation Guide](#-installation-guide) section below. If you wish to use the previous release version of OpenVINO, please checkout the [2024.6 branch](https://github.com/openvinotoolkit/openvino_notebooks/tree/2024.5). If you wish to use the previous Long Term Support (LTS) version of OpenVINO check out the [2023.3 branch](https://github.com/openvinotoolkit/openvino_notebooks/tree/2023.3).

If you need help, please start a GitHub [Discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions).  


If you run into issues, please check the [troubleshooting section](#-troubleshooting), [FAQs](#-faq) or start a GitHub [discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions).

Notebooks with ![binder logo](https://mybinder.org/badge_logo.svg) and ![colab logo](https://colab.research.google.com/assets/colab-badge.svg) buttons can be run without installing anything. [Binder](https://mybinder.org/) and [Google Colab](https://colab.research.google.com/) are free online services with limited resources. For the best performance, please follow the [Installation Guide](#-installation-guide) and run the notebooks locally.

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-system-requirements'></div>

## ⚙️ System Requirements

The notebooks run almost anywhere &mdash; your laptop, a cloud VM, or even a Docker container. The table below lists the supported operating systems and Python versions.

| Supported Operating System                                 | [Python Version (64-bit)](https://www.python.org/) |
| :--------------------------------------------------------- |:---------------------------------------------------|
| Ubuntu 20.04 LTS, 64-bit                                   | 3.9 - 3.12                                         |
| Ubuntu 22.04 LTS, 64-bit                                   | 3.9 - 3.12                                         |
| Red Hat Enterprise Linux 8, 64-bit                         | 3.9 - 3.12                                         |
| CentOS 7, 64-bit                                           | 3.9 - 3.12                                         |
| macOS 10.15.x versions or higher                           | 3.9 - 3.12                                         |
| Windows 10, 64-bit Pro, Enterprise or Education editions   | 3.9 - 3.12                                         |
| Windows Server 2016 or higher                              | 3.9 - 3.12                                         |

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)](#)
<div id='-run-the-notebooks'></div>

## 💻 Run the Notebooks

### To Launch a Single Notebook

If you wish to launch only one notebook, like the Monodepth notebook, run the command below (from the repository root directory):

```bash
jupyter lab notebooks/vision-monodepth/vision-monodepth.ipynb
```

### To Launch all Notebooks

Launch Jupyter Lab with index `README.md` file opened for easier navigation between notebooks directories and files. Run the following command from the repository root directory:
	
```bash
jupyter lab notebooks/README.md
```

Alternatively, in your browser select a notebook from the file browser in Jupyter Lab using the left sidebar. Each tutorial is located in a subdirectory within the `notebooks` directory.

<img src="https://user-images.githubusercontent.com/15709723/120527271-006fd200-c38f-11eb-9935-2d36d50bab9f.gif">

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-cleaning-up'></div>

## 🧹 Cleaning Up

<div id='-shut-down-jupyter-kernel' markdown="1">

1. Shut Down Jupyter Kernel

	To end your Jupyter session, press `Ctrl-c`. This will prompt you to `Shutdown this Jupyter server (y/[n])?` enter `y` and hit `Enter`.
</div>	
	
<div id='-deactivate-virtual-environment' markdown="1">

2. Deactivate Virtual Environment

	To deactivate your virtualenv, simply run `deactivate` from the terminal window where you activated `openvino_env`. This will deactivate your environment.

	To reactivate your environment, run `source openvino_env/bin/activate` on Linux or `openvino_env\Scripts\activate` on Windows, then type `jupyter lab` or `jupyter notebook` to launch the notebooks again.
</div>	

<div id='-delete-virtual-environment' markdown="1">

3. Delete Virtual Environment _(Optional)_

	To remove your virtual environment, simply delete the `openvino_env` directory:
</div>

<div id='-on-linux-and-macos' markdown="1">

  - On Linux and macOS:

	```bash
	rm -rf openvino_env
	```
</div>

<div id='-on-windows' markdown="1">

  - On Windows:

	```bash
	rmdir /s openvino_env
	```
</div>

<div id='-remove-openvino-env-kernel' markdown="1">

  - Remove `openvino_env` Kernel from Jupyter

	```bash
	jupyter kernelspec remove openvino_env
	```
</div>

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-troubleshooting'></div>

## ⚠️ Troubleshooting

If these tips do not solve your problem, please open a [discussion topic](https://github.com/openvinotoolkit/openvino_notebooks/discussions)
or create an [issue](https://github.com/openvinotoolkit/openvino_notebooks/issues)!

- To check some common installation problems, run `python check_install.py`. This script is located in the openvino_notebooks directory.
  Please run it after activating the `openvino_env` virtual environment.
- If you get an `ImportError`, double-check that you installed the Jupyter kernel. If necessary, choose the `openvino_env` kernel from the _Kernel->Change Kernel_ menu in Jupyter Lab or Jupyter Notebook.
- If OpenVINO is installed globally, do not run installation commands in a terminal where `setupvars.bat` or `setupvars.sh` are sourced.
- For Windows installation, it is recommended to use _Command Prompt (`cmd.exe`)_, not _PowerShell_.
- If you get `ImportError: cannot import name 'collect_telemetry' from 'notebook_utils'`, make sure that you have the latest version of `notebook_utils.py` file downloaded in the notebook directory. Try removing outdated `notebook_utils.py` file and re-run the notebook - new utils file will be downloaded.

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-telemetry'></div>

## 📊 Telemetry

When you execute a notebook cell that contains `collect_telemetry()` function, telemetry data is collected to help us improve your experience.
This data only indicates that the cell was executed and does **not** include any personally identifiable information (PII).

By default, anonymous telemetry data is collected, limited solely to the execution of the notebook.
This telemetry does **not** extend to any Intel software, hardware, websites, or products.

If you prefer to disable telemetry, you can do so at any time by commenting out the specific line responsible for data collection in the notebook:
```python
# collect_telemetry(...)
```
Also you can disable telemetry collection by setting `SCARF_NO_ANALYTICS` or `DO_NOT_TRACK` environment variable to `1`:
```bash
export SCARF_NO_ANALYTICS=1
# or
export DO_NOT_TRACK=1
```

Scarf is used for telemetry purposes. Refer to [Scarf documentation](https://docs.scarf.sh/) to understand how the data is collected and processed.

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-additional-resource'></div>

## 📚 Additional Resources
* [OpenVINO Blog](https://blog.openvino.ai/) - a collection of technical articles with OpenVINO best practices, interesting use cases and tutorials.
* [Awesome OpenVINO](https://github.com/openvinotoolkit/awesome-openvino) - a curated list of OpenVINO based AI projects.
* [OpenVINO GenAI Samples](https://github.com/openvinotoolkit/openvino.genai?tab=readme-ov-file#openvino-genai-samples) - collection of OpenVINO GenAI API samples.
* [Edge AI Reference Kit](https://github.com/openvinotoolkit/openvino_build_deploy) - pre-built components and code samples designed to accelerate the development and deployment of production-grade AI applications across various industries, such as retail, healthcare, and manufacturing.
* [Open Model Zoo demos](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md) -  console applications that provide templates to help implement specific deep learning inference scenarios. These applications show how to preprocess and postprocess data for model inference and organize processing pipelines.
* [oneAPI-samples](https://github.com/oneapi-src/oneAPI-samples) repository demonstrates the performance and productivity offered by oneAPI and its toolkits such as oneDNN in a multiarchitecture environment. OpenVINO™ toolkit takes advantage of the discrete GPUs using oneAPI, an open programming model for multi-architecture programming.

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)](#-contributors)
<div id='-contributors'></div>

## 🧑‍💻 Contributors

<a href="https://github.com/openvinotoolkit/openvino_notebooks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openvinotoolkit/openvino_notebooks" />
</a>

Made with [`contrib.rocks`](https://contrib.rocks).

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-faq'></div>

## ❓ FAQ

* [Which devices does OpenVINO support?](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes.html)
* [What is the first CPU generation you support with OpenVINO?](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html)
* [Are there any success stories about deploying real-world solutions with OpenVINO?](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html)

---

\* Other names and brands may be claimed as the property of others.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=README.md" />

Human Rights Information: “Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See Intel’s Global Human Rights Principles at https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf. Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.