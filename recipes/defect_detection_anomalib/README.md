# Defect Detection wit Anomalib

Product defects are a significant issue for manufacturers, leading to wasted resources, decreased customer satisfaction, and reduced profits. Despite decades of efforts to implement quality control measures, manufacturers have struggled to eliminate product defects entirely due to various challenges. 

Intel’s OpenVINO™ Defect Detection wit Anomalib offers a comprehensive solve to their quality control problem by providing companies and their technical teams with a single-source end-to-end solution to catch manufacturing defects in real-time. The kit trains a model to detect defects using Anomalib, is an open-source deep learning library that makes it easy to train, test and deploy different anomaly detection algorithms on both public and custom datasets. The model can be exported to the OpenVINO™ Intermediate Representation and deployed on Intel hardware. That are optimized for inference performance, trainable on CPU, and require low memory use, making it suitable for deployment on the edge. The kit also includes educational documentation and resources to empower developers to make decisions with confidence every step of the way. By streamlining the process of researching and implementing quality control measures, this recipe kit enables companies and their developers to solve this real-world problem while reducing costs, time, and risks.

This AI Recipe use the notebooks in the actual Anomalib repository. Here you will find that repository as a submodule.

| Notebook                       |                                                                                                                                                                                                                                                          |     |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| Dataset Creation and Inference | [![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501a_dataset_creation_and_Inference_with_a_robotic_arm.ipynb) |
| Training a Model               | [![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501b_training_a_model_with_cubes_from_a_robotic_arm.ipynb)    |

Before to run the above notebooks run the installation process inside of OpenVINO Notebooks environment.

## Installation Instructions

If you have not installed all required dependencies, just run `pip install anomalib` in the same OpenVINO Notebooks environment.

## Notebook Contents

This notebook demonstrates how NNCF can be used to compress a model trained with Anomalib. The notebook is divided into the following sections:

- Train an anomalib model without compression
- Train a model with NNCF compression
- Compare the performance of the two models (FP32 vs INT8)

Step 1: Then connect your USB Camera and verify it works using a simple camera application. Once it is verified, close the application.

Step 2 (Optional): If you have the Dobot robot please make the following.

1. Install Dobot requirements (See Dobot documentation here: https://en.dobot.cn/products/education/magician.html).
2. Check all connections to the Dobot and verify it is working using the Dobot Studio.
3. Install the vent accessory on the Dobot and verify it is working using Dobot Studio.
4. In the Dobot Studio, hit the "Home" button, and locate the:

![image](https://user-images.githubusercontent.com/10940214/219142393-c589f275-e01a-44bb-b499-65ebeb83a3dd.png)

a. Calibration coordinates: Initial position upper-left corner of cubes array.

b. Place coordinates: Position where the arm should leave the cubic over the conveyor belt.

c. Anomaly coordinates: Where you want to release the abnormal cube.

d. Then, replace those coordinates in the notebook

### Data acquisition and inferencing

For data acquisition and inferencing we will use [501a notebook](https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501a_dataset_creation_and_Inference_with_a_robotic_arm.ipynb). There we need to identify the `acquisition` flag, **True** for _acquisition mode_ and **False** for _inferencing mode_. In acquisition mode be aware of the _normal_ or _abnormal_ folder we want to create, in this mode the notebook will save every image in the anomalib/datasets/cubes/{FOLDER} for further training. In inferencing mode the notebook won't save images, it will run the inference and show the results.

_Note_: If you dont have the robot you could jump to the another notebook [501b](https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/500_use_cases/501_dobot/501b_training_a_model_with_cubes_from_a_robotic_arm.ipynb) and download the dataset from this [link](https://github.com/openvinotoolkit/anomalib/releases/tag/dobot)

### Training

For training we will use [501_2 notebook](https://github.com/openvinotoolkit/anomalib/blob/feature/notebooks/usecases/dobot/notebooks/500_use_cases/dobot/501_2_Training%20a%20model%20with%20cubes%20from%20a%20robotic%20arm.ipynb). In this example we are using "Padim" model and we are using Anomalib API for setting up the dataset, model, metrics, and the optimization process with OpenVINO.

### Have Fun and share your results in the discussion channel! 😊
