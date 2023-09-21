# TensorFlow Hub models + OpenVINO

This tutorial demonstrates how to convert TensorFlow Hub models to OpenVINO Intermediate Representation.

[TensorFlow Hub](https://tfhub.dev/) is a library and online platform developed by Google that simplifies machine learning model reuse and sharing. It serves as a repository of pre-trained models, embeddings, and reusable components, allowing researchers and developers to access and integrate state-of-the-art machine learning models into their own projects with ease. TensorFlow Hub provides a diverse range of models for various tasks like image classification, text embedding, and more. It streamlines the process of incorporating these models into TensorFlow workflows, fostering collaboration and accelerating the development of AI applications. This centralized hub enhances model accessibility and promotes the rapid advancement of machine learning capabilities across the community.

## Image classification
### Description
This tutorial demonstrates step-by-step instructions on how to do inference on a classification model loaded from TensorFlow Hub using OpenVINO Runtime.

We will use the [MobileNet_v2](https://arxiv.org/abs/1704.04861) model from [TensorFlow Hub](https://tfhub.dev/) to demonstrate how to convert TensorFlow models to OpenVINO Intermediate Representation.

MobileNetV2 is a compact and efficient deep learning architecture designed for mobile and embedded devices, developed by Google researchers. It builds on the success of the original MobileNet by introducing improvements in both speed and accuracy. MobileNetV2 employs a streamlined architecture with inverted residual blocks, making it highly efficient for real-time applications while minimizing computational resources. This network excels in tasks like image classification, object detection, and image segmentation, offering a balance between model size and performance. MobileNetV2 has become a popular choice for on-device AI applications, enabling faster and more efficient deep learning inference on smartphones and edge devices.

### Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).