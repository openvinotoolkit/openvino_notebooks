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

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/model-server/README.md" />
