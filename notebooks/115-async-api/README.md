# Runtime Inference Optimization with Async API

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F115-async-api%2F115-async-api.ipynb)


This notebook demostrate how to use [asynchronous execution](https://docs.openvino.ai/nightly/openvino_docs_deployment_optimization_guide_common.html) to improve data pipelining. 

OpenVINO™ Runtime supports inference in either synchronous or asynchronous mode. Using the Async API can improve application’s overall frame-rate, because rather than wait for inference to complete, the app can keep working on the host, while the accelerator is busy. 


## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.