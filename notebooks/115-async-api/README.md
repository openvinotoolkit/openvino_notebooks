# Asynchronous Inference with OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F115-async-api%2F115-async-api.ipynb)


This notebook demostrate how to use [asynchronous execution](https://docs.openvino.ai/latest/openvino_docs_deployment_optimization_guide_common.html) to improve data pipelining. 

OpenVINO™ Runtime supports inference in either synchronous or asynchronous mode. Using the Async API can improve application’s overall frame-rate, because thie function can keep the hardware busy by sending inference requests continuously. In other words, if you can afford sending a new request without waiting for the previous one to finish. 

With synchronous mode, we wait for the result of the first inference before sending the next request. While the request is being sent, the hardware is idle. When we use the async API, the transfer of the second request is overlapped with the execution of the first inference, and that prevents any hardware idle time. ​

![async vs sync](https://user-images.githubusercontent.com/91237924/180628033-514f4475-8a55-44a0-a22e-73aa70d8868f.png)

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
