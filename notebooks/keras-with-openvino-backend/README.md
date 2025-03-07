# Run inference in Keras 3 with the OpenVINO™ IR backend

Starting with release 3.8, [Keras](https://github.com/keras-team/keras) provides native integration with the OpenVINO backend for accelerated inference. This integration enables you to leverage OpenVINO performance optimizations directly within the Keras workflow, enabling faster inference on OpenVINO supported hardware.


In this tutorial, we will show how to run inference of an end-to-end [BERT model for classification tasks](https://www.kaggle.com/models/keras/bert/) using the OpenVINO backend.


>**Note**: The OpenVINO backend may currently lack support for some operations. This will be addressed in upcoming Keras releases as operation coverage is being expanded.


## Notebook contents
- Prerequisites
- Load the model with the OpenVINO backend and inference
- Sentiment Classification Example


## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/keras-with-openvino-backend/README.md" />
