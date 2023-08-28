# People Counting with OpenVINO™

Here are the steps involved in this demo:

Step 0: Install Python and prerequisites

Step 1: Set up the environment

Step 2: Convert and Optimize the YOLOv8 Model

Step 3: Run the Application

Now, let's dive into the steps starting with installing Python. We recommend using Ubuntu to set up and run this project.

## Step 0

This project requires Python 3.8 or higher and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

```shell
sudo apt install git git-lfs gcc python3-venv python3-dev
```

_NOTE: If you are using Windows, you will probably need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) also._

## Step 1

1. Clone the Repository

To clone the repository, run the following command:

```shell
git clone -b conference-demos https://github.com/openvinotoolkit/openvino_notebooks.git openvino_notebooks
```

The above will clone the repository into a directory named "openvino_notebooks" in the current directory. It will also download a sample video. Then, navigate into the directory using the following command:

```shell
cd openvino_notebooks/conference-demos/people-counter-demo
```

2. Create a virtual environment

To create a virtual environment, open your terminal or command prompt and navigate to the directory where you want to create the environment. Then, run the following command:

```shell
python3 -m venv venv
```
This will create a new virtual environment named "venv" in the current directory.

3. Activate the environment

Activate the virtual environment using the following command:

```shell
source venv/bin/activate   # For Unix-based operating system such as Linux or macOS
```

_NOTE: If you are using Windows, use `venv\Scripts\activate` command instead._

This will activate the virtual environment and change your shell's prompt to indicate that you are now working within that environment.

4. Install the Packages

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

## Step 2

To convert and optimize the YOLOv8 model, run the [convert-and-optimize-the-model.ipynb](convert-and-optimize-the-model.ipynb) notebook. This will convert the YOLOv8 model to an OpenVINO™ Intermediate Representation (IR) format and optimize it for use with OpenVINO™.

## Step 3

To run the application, use the [run-the-application.ipynb](run-the-application.ipynb). To stop the application please, press 'q' or escape at any time. There are some arguments in the last cell, which you can change to change the behavior of the app.

Congratulations! You have successfully set up and run the People Counting application with OpenVINO™.
