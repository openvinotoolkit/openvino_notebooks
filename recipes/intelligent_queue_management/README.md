# Intelligent Queue Management with OpenVINO™

The Intelligent Queue Management project is an application that uses OpenVINO™, a toolkit that enables developers to deploy deep learning models on a variety of hardware platforms. The application is designed to help businesses manage customer queues more effectively, by analyzing video streams from cameras and detecting the number of people in each queue. The system then uses this information to optimize the queuing process and reduce waiting times for customers.

Here are the steps involved in this project:

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
git clone -b recipes https://github.com/openvinotoolkit/openvino_notebooks.git openvino_notebooks
```

The above will clone the repository into a directory named "openvino_notebooks" in the current directory. It will also download a sample video. Then, navigate into the directory using the following command:

```shell
cd openvino_notebooks/recipes/intelligent_queue_management
```

Then pull video sample:

```shell
git lfs pull
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
python convert_and_optimize.py --model_name yolov8m --model_dir model --data_dir data --quantize True
```
This will convert the YOLOv8 model to an OpenVINO™ Intermediate Representation (IR) format and optimize it for use with OpenVINO™.
You can run either the python script or check out [convert-and-optimize-the-model.ipynb](docs/convert-and-optimize-the-model.ipynb) to learn more.

## Step 3

To run the application, use the following command:

```shell
python app.py --stream sample_video.mp4 --model_path model/yolov8m_openvino_int8_model/yolov8m.xml --zones_config_file zones.json --customers_limit 3
```
This will run the application with the specified arguments. Replace "video_file.mp4" with the path to your input video file, "zones.json" with the path to your zones configuration file, and "3" with the maximum number of customers allowed in the queue.
You can also run the [run-the-application.ipynb](docs/run-the-application.ipynb) to learn more about the inference process. To stop the application please, press 'q' or escape at any time.

_NOTE: Alternatively, you can run all steps with the following command:_

```shell
python main.py --stream sample_video.mp4
```

## Step 4

Benchmark the Model with OpenVINO's Benchmark_App
Benchmarking provides insight into your YOLOv8 model's real-world performance. Performance may vary based on use and configuration.

### Benchmark Results 

![YOLOv8m Benchmark Results](https://github.com/AnishaUdayakumar/intelligent-queue-management-openvino/assets/109281183/8a81243e-ee32-4b30-9994-326ecb07d32f)

The benchmarks for the YOLOv8m model were run using OpenVINO version 2023.0.0 on the 4th Gen Intel® Xeon® Scalable Processor:

* **Throughput** 
  * FP16: 252FPS
  * INT8: 195 FPS
* **Latency**
  * FP16: 7.94 ms
  * INT8: 10.19 ms

These figures represent the model's theoretical maximums. It's recommended to test the model in your deployment environment to gauge its real-world performance.

### Running the Benchmark

Use the following command to run the benchmark:

```shell
!benchmark_app -m $int8_model_det_path -d $device -hint latency -t 30
```
Replace `int8_model_det_path` with the path to your INT8 model and $device with the specific device you're using (CPU, GPU, etc.). This command performs inference on the model for 30 seconds. Run `benchmark_app --help` for additional command-line options.

Congratulations! You have successfully set up, run, and benchmarked the Intelligent Queue Management application with OpenVINO™.
