# Virtual Medical Assistant with OpenVINO™

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/LICENSE)

The Virtual Medical Assistant project is an application... _(TODO)_

## Getting Started

Now, let's dive into the steps starting with installing Python. We recommend using Ubuntu to set up and run this project.

### Installing Prerequisites

This project requires Python 3.8 or higher and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

```shell
sudo apt install git gcc python3-venv python3-dev
```

_NOTE: If you are using Windows, you will probably need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) also._

### Setting up your Environment

#### Cloning the Repository

To clone the repository, run the following command:

```shell
git clone -b recipes https://github.com/openvinotoolkit/openvino_notebooks.git openvino_notebooks
```

The above will clone the repository into a directory named "openvino_notebooks" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_notebooks/recipes/virtual_medical_assistant
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

_NOTE: If you are using Windows, use `venv\Scripts\activate` command instead._

This will activate the virtual environment and change your shell's prompt to indicate that you are now working within that environment.

#### Installing the Packages

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

### Converting and Optimizing the Models

To download the LLama2 model from the Hugging Face Hub you need to accept the license. 

_TODO (instructions needed here)_

Then you have to be authenticated, so run the following command and provide your access token available under your profile [settings](https://huggingface.co/settings/tokens). You don't have to add it as git credentials.
```shell
huggingface-cli login
```

Now, you're ready to run the script.
```shell
python convert_and_optimize.py
```

_NOTE: Running the above script may take up to 120 minutes (depending on your hardware Internet connection), as the models are huge._
