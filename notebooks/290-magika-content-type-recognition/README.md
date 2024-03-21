# Magika: AI powered fast and efficient file type identification using OpenVINO

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2F290-magika-content-type-recognition%2F290-magika-content-type-recognition.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/290-magika-content-type-recognition/290-magika-content-type-recognition.ipynb)

Magika is a novel AI powered file type detection tool that relies on the recent advance of deep learning to provide accurate detection. Under the hood, Magika employs a custom, highly optimized model that only weighs about 1MB, and enables precise file identification within milliseconds, even when running on a single CPU.


**Why identifying file type is difficult?**

Since the early days of computing, accurately detecting file types has been crucial in determining how to process files. Linux comes equipped with `libmagic` and the `file` utility, which have served as the de facto standard for file type identification for over 50 years. Today web browsers, code editors, and countless other software rely on file-type detection to decide how to properly render a file. For example, modern code editors use file-type detection to choose which syntax coloring scheme to use as the developer starts typing in a new file.

Accurate file-type detection is a notoriously difficult problem because each file format has a different structure, or no structure at all. This is particularly challenging for textual formats and programming languages as they have very similar constructs. So far, `libmagic` and most other file-type-identification software have been relying on a handcrafted collection of heuristics and custom rules to detect each file format.

This manual approach is both time consuming and error prone as it is hard for humans to create generalized rules by hand. In particular for security applications, creating dependable detection is especially challenging as attackers are constantly attempting to confuse detection with adversarially-crafted payloads.

To address this issue and provide fast and accurate file-type detection Magika was developed. More details about approach and model can be found in original [repo](https://github.com/google/magika) and [Google's blog post](https://opensource.googleblog.com/2024/02/magika-ai-powered-fast-and-efficient-file-type-identification.html).

In this tutorial we consider how to bring OpenVINO power into Magika.

## Notebook Contents

This tutorial demonstrates step-by-step instructions on how to launch Magika file type identification tool powered by OpenVINO. The tutorial consists of following parts:

- Define model loading interface for OpenVINO
- Create model instance
- Run OpenVINO model inference on bytes input
- Run OpenVINO model inference on file path input
- Launch interactive demo 


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).