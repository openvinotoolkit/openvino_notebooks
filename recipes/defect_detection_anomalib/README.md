# Defect Detection with Anomalib

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/LICENSE)

Intel’s OpenVINO™ Defect Detection with Anomalib offers a comprehensive solution to their quality control problem by providing companies and their technical teams with a single-source end-to-end solution to catch manufacturing defects in real time. 

This AI Recipe uses the notebooks in the actual Anomalib repository. Here you will find that repository as a submodule.

## Table of Contents

- [Installing Anomalib](#installing-anomalib)
- [Getting Started with the Jupyter Notebook](#getting-started-with-the-jupyter-notebook)
	- [Setting up your camera](#setting-up-your-camera)
	- [Setting up the Dobot Robot (Optional)](#setting-up-the-dobot-robot-optional)
	- [Data Acquisition and Inferencing](#data-acquisition-and-inferencing)
- [Training](#training)
- [Understanding Defect Detection](#understanding-defect-detection)
- [Troubleshooting and Resources](#troubleshooting-and-resources)

| Notebook |  |
| - | - |
| Training a Model | [![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501a_training_a_model_with_cubes_from_a_robotic_arm.ipynb) |
| Dataset Creation and Inference | [![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501b_inference_with_a_robotic_arm.ipynb) |

Before running the above notebooks, run the installation process inside the OpenVINO Notebooks environment.

## Installing Anomalib

If you have not installed all required dependencies, just run `pip install anomalib` in the same OpenVINO Notebooks environment.

## Getting Started with the Jupyter Notebook

This notebook demonstrates how NNCF can be used to compress a model trained with Anomalib. The notebook is divided into the following sections:

- Train an Anomalib model without compression
- Train a model with NNCF compression
- Compare the performance of the two models (FP32 vs INT8)

### Setting up your Camera

Connect your USB Camera and verify it works using a simple camera application. Once it is verified, close the application.

### Setting up the Dobot Robot (Optional)

1. Install Dobot requirements (See Dobot documentation here: https://en.dobot.cn/products/education/magician.html).
2. Check all connections to the Dobot and verify it is working using the Dobot Studio.
3. Install the vent accessory on the Dobot and verify it is working using Dobot Studio.
4. In the Dobot Studio, hit the "Home" button, and locate the:

![image](https://user-images.githubusercontent.com/10940214/219142393-c589f275-e01a-44bb-b499-65ebeb83a3dd.png)

a. Calibration coordinates: Initial position upper-left corner of cubes array.

b. Place coordinates: Position where the arm should leave the cubic over the conveyor belt.

c. Anomaly coordinates: Where you want to release the abnormal cube.

d. Then, replace those coordinates in the notebook

### Data Acquisition and Inferencing

For data acquisition and inferencing we will use [501b notebook](https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501b_inference_with_a_robotic_arm.ipynb). There we need to identify the `acquisition` flag, **True** for _acquisition mode_ and **False** for _inferencing mode_. In acquisition mode be aware of the _normal_ or _abnormal_ folder we want to create, in this mode the notebook will save every image in the anomalib/datasets/cubes/{FOLDER} for further training. In inferencing mode the notebook won't save images, it will run the inference and show the results.

_Note_: If you don't have the robot you could jump to another notebook [501a](https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501a_training_a_model_with_cubes_from_a_robotic_arm.ipynb) and download the dataset from this [link](https://github.com/openvinotoolkit/anomalib/releases/tag/dobot)

### Training

For training, we will use the [501a notebook](https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501a_training_a_model_with_cubes_from_a_robotic_arm.ipynb). In this example we are using "Padim" model and we are using Anomalib API for setting up the dataset, model, metrics, and the optimization process with OpenVINO.

## Understanding Defect Detection

Product defects are a significant issue for manufacturers, leading to wasted resources, decreased customer satisfaction, and reduced profits. Despite decades of efforts to implement quality control measures, manufacturers have struggled to eliminate product defects entirely due to various challenges. 

The kit trains a model to detect defects using Anomalib, an open-source deep-learning library, to make it easy to train, test, and deploy different anomaly detection algorithms on both public and custom datasets. The model can be exported to the OpenVINO™ Intermediate Representation and deployed on Intel hardware. They are optimized for inference performance, trainable on CPU, and require low memory use, making them suitable for deployment on the edge.

The kit also includes educational documentation and resources to empower developers to make decisions with confidence every step of the way. By streamlining the process of researching and implementing quality control measures, this recipe kit enables companies and their developers to solve this real-world problem while reducing costs, time, and risks.

## Troubleshooting and Resources
 Have Fun and share your results in the discussion channel! 😊

- Open a [discussion topic](https://github.com/openvinotoolkit/openvino_notebooks/discussions)
- Create an [issue](https://github.com/openvinotoolkit/openvino_notebooks/issues)
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2023.0/home.html)
