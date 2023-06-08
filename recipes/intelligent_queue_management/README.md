# Intelligent Queue Management with OpenVINO™

The Intelligent Queue Management project is an application that uses OpenVINO™, a toolkit that enables developers to deploy deep learning models on a variety of hardware platforms. The application is designed to help businesses manage customer queues more effectively, by analyzing video streams from cameras and detecting the number of people in each queue. The system then uses this information to optimize the queuing process and reduce waiting times for customers.

Here are the steps involved in this project:

Step 0: Install Python

Step 1: Set up the environment

Step 2: Convert and Optimize the YOLOv8 Model

Step 3: Run the Application

Now, let's dive into the steps starting with installing Python.

## Step 0

This project requires Python 3.8 or higher. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

## Step 1

1. Clone the Repository

To clone the repository, run the following command:

```shell
git clone -b recipes https://github.com/openvinotoolkit/openvino_notebooks.git openvino_notebooks
```

The above will clone the repository into a directory named "openvino_notebooks" in the current directory. It will also download a sample video. Then, navigate into the directory using the following command:

```shell
cd openvino_notebooks/recipes/intelligent_queue_management
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

To convert and optimize the YOLOv8 model, run the following command:

```shell
python convert_and_optimize.py --model_name yolov8m --model_dir model --data_dir data --quantize
```
This will convert the YOLOv8 model to an OpenVINO™ Intermediate Representation (IR) format and optimize it for use with OpenVINO™.
You can run either the python script or check out [convert-and-optimize-the-model.ipynb](docs/convert-and-optimize-the-model.ipynb) to learn more.

## Step 3

To run the application, use the following command:

```shell
python app.py --stream sample_video.mp4 --model_path model/yolov8m_openvino_int8_model/yolov8m.xml --zones_config_file zones.json --customers_limit 3
```
This will run the application with the specified arguments. Replace "video_file.mp4" with the path to your input video file, "zones.json" with the path to your zones configuration file, and "3" with the maximum number of customers allowed in the queue.
You can also run the [run-the-application.ipynb](docs/run-the-application.ipynb) to learn more about the inference process.

_NOTE: Alternatively, you can run all steps with the following command:_

```shell
python main.py --stream sample_video.mp4
```

Congratulations! You have successfully set up and run the Intelligent Queue Management application with OpenVINO™.
