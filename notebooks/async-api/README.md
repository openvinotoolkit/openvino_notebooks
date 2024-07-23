# Asynchronous Inference with OpenVINO™

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Fasync-api%2Fasync-api.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/async-api/async-api.ipynb)


This notebook demonstrates how to use the [Async API](https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/general-optimizations.html) and [`AsyncInferQueue`](https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application/python-api-exclusives.html#asyncinferqueue) for asynchronous execution with OpenVINO.

OpenVINO Runtime supports inference in either synchronous or asynchronous mode. The key advantage of the Async API is that when a device is busy with inference, the application can perform other tasks in parallel (for example, populating inputs or scheduling other requests) rather than wait for the current inference to complete first.

With synchronous mode, we wait for the result of the first inference before sending the next request. While the request is being sent, the hardware is idle. When we use the async API, the transfer of the second request is overlapped with the execution of the first inference, and that prevents any hardware idle time. ​

![async vs sync](https://user-images.githubusercontent.com/91237924/180628033-514f4475-8a55-44a0-a22e-73aa70d8868f.png)

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/async-api/README.md" />
