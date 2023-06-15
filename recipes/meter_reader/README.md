# Automatic Industrial Meter Reading with OpenVINO™

![workflow](https://user-images.githubusercontent.com/91237924/166137115-67284fa5-f703-4468-98f4-c43d2c584763.png)

The automatic industrial meter reading project is an application that uses OpenVINO™, a toolkit that enables developers to deploy deep learning models on a variety of hardware platforms. The application is designed to help .

Here are the steps involved in this project:

Step 0: Install Python

Step 1: Set up the environment

Step 2: Run the Application

Now, let's dive into the steps starting with installing Python.

## Step 0

This project requires Python 3.7 or higher. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

## Step 1

1. Clone the Repository

To clone the repository, run the following command:

```shell
git clone -b recipes https://github.com/openvinotoolkit/openvino_notebooks.git openvino_notebooks
```

This will clone the repository into a directory named "meter-reader-openvino" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_notebooks/recipes/meter_reader
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

NOTE: If you are using Windows, use `venv\Scripts\activate` command instead.

This will activate the virtual environment and change your shell's prompt to indicate that you are now working within that environment.

4. Install the Packages

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```


## Step 2

1. Prepare your detection and segmentation models:
```shell
cd model
sudo sh ./download_pdmodel.sh
```

2. To run the application, use the following command:

```shell
python main.py -i data/test.jpg -c config/config.json  -t "analog"
```

This will run the application with the specified arguments. Replace "data/test.jpg" with the path to your input image.
The result images will be exported to "data" fold. You can also run the [run-the-application.ipynb](docs/run-the-application.ipynb) to learn more about the inference process.

Congratulations! You have successfully set up and run the Automatic Industrial Meter Reading application with OpenVINO™.
