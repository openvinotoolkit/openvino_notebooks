# Smart Meter Scanning with OpenVINO™ Toolkit

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/LICENSE)

Smart Meter Scanning is an AI-based application that enables cameras to automatically read results from your analog meters, transforming it into digital data with accurate, near-real-time meter results. It uses computer vision, object detection, and object segmentation.

![workflow](https://user-images.githubusercontent.com/91237924/166137115-67284fa5-f703-4468-98f4-c43d2c584763.png)

## Table of Contents

- [Getting Started](#getting-started)
	- [Installing Prerequisites](#installing-prerequisites)
	- [Setting up your Environment](#setting-up-your-environment)
		- [Cloning the Repository](#cloning-the-repository)
		- [Creating a Virtual Environment](#creating-a-virtual-environment)
		- [Activating the Environment](#activating-the-environment)
		- [Installing the Packages](#installing-the-packages)
	- [Preparing your Models](#preparing-your-models)
	- [Running the Application](#running-the-application)
- [Troubleshooting and Resources](#troubleshooting-and-resources)

## Getting Started

Now, let's dive into the steps, starting with installing Python. 

### Installing Prerequisites

This project requires Python 3.7 or higher. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

### Setting up your Environment

#### Cloning the Repository

To clone the repository, run the following command:

```shell
git clone -b recipes https://github.com/openvinotoolkit/openvino_notebooks.git openvino_notebooks
```

This will clone the repository into a directory named "meter-reader-openvino" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_notebooks/recipes/meter_reader
```

#### Creating a Virtual Environment

To create a virtual environment, open your terminal or command prompt and navigate to the directory where you want to create the environment. Then, run the following command:

```shell
python3 -m venv venv
```
This will create a new virtual environment named "venv" in the current directory.

#### Activating the Environment

Activate the virtual environment using the following command:

```shell
source venv/bin/activate   # For Unix-based operating system such as Linux or macOS
```

NOTE: If you are using Windows, use `venv\Scripts\activate` command instead.

This will activate the virtual environment and change your shell's prompt to indicate that you are now working within that environment.

#### Installing the Packages

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

### Preparing your Models 

Prepare your detection and segmentation models with this command: 
```shell
cd model
sudo sh ./download_pdmodel.sh
```

### Running the Application

To run the application, use the following command:

```shell
python main.py -i data/test.jpg -c config/config.json  -t "analog"
```

This will run the application with the specified arguments. Replace "data/test.jpg" with the path to your input image.
The result images will be exported to same fold of test image. You can also run the [203-meter-reader.ipynb](../../notebooks/203-meter-reader/203-meter-reader.ipynb) to learn more about the inference process.

Congratulations! You have successfully set up and run the Automatic Industrial Meter Reading application with OpenVINO™.

## Troubleshooting and Resources
- Open a [discussion topic](https://github.com/openvinotoolkit/openvino_notebooks/discussions)
- Create an [issue](https://github.com/openvinotoolkit/openvino_notebooks/issues)
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2023.0/home.html)
