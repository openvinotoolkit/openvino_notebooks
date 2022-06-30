 # PaddlePaddle to ONNX and OpenVINO™ IR Tutorial

 [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F103-paddle-onnx-to-openvino%2F103-paddle-onnx-to-openvino-classification.ipynb) 

![PaddlePaddle Classification](https://user-images.githubusercontent.com/77325899/127503530-72c8ce57-ef6f-40a7-808a-d7bdef909d11.png)

This notebook shows how to convert [PaddlePaddle](https://www.paddlepaddle.org.cn) models to OpenVINO IR.

## Notebook Contents

The notebook uses [Paddle2ONNX](https://github.com/PaddlePaddle/paddle2onnx) and OpenVINO [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) to convert a MobileNet V3 [PaddleHub](https://github.com/PaddlePaddle/PaddleHub) model, pre-trained on the [ImageNet](https://www.image-net.org) dataset, to OpenVINO IR. It also shows how to perform classification inference on an image, using [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html) and compares the results of the PaddlePaddle model with the OpenVINO IR model. 

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
