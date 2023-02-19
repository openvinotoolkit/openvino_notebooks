# Introduction to OpenVINO™ Model Server

This notebook demonstrates how to deploy a model server and request predictions from a client application.

OpenVINO Model Server (OVMS) is a high-performance system for serving models. Implemented in C++ for scalability and optimized for deployment on Intel architectures, the model server uses the same architecture and API as TensorFlow Serving and KServe while applying OpenVINO for inference execution. Inference service is provided via gRPC or REST API, making deploying new algorithms and AI experiments easy.

![ovms_high_level](https://user-images.githubusercontent.com/91237924/215658767-0e0fc221-aed0-4db1-9a82-6be55f244dba.png)

## Notebook Contents

The notebook covers following steps:

* Prepare Docker
* Preparing a Model Repository
* Start the Model Server Container
* Prepare the Example Client Components

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
