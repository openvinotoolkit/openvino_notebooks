# PyTorch to ONNX and OpenVINO™ IR Tutorial


This notebook demonstrates how to do inference on Real-ESRGAN a PyTorch image restoration model for real images, using [OpenVINO](https://github.com/openvinotoolkit/openvino).

# Example

Low-resolution image:
![low-res]()


4x-scaled enhanced image
![scaled-res]()


## Notebook Contents

The notebook uses [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) to convert the open-source Real-ESRGAN image restoration model from [pytorch model](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md) to OpenVINO IR. It also shows comparisons between pytorch, [ONNX](https://onnxruntime.ai/docs/execution-providers/), and [OpenVINO runtime](https://docs.openvino.ai/latest/openvino_docs_OV_UG_OV_Runtime_User_Guide.html). 

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
