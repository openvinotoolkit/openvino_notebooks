# The attention center model with OpenVINO™

This notebook demonstrates how the use [attention center model](https://github.com/google/attention-center/tree/main) with OpenVINO. This model is in the [TensorFlow Lite format](https://www.tensorflow.org/lite), which is supported in OpenVINO now by TFlite frontend. Check out [this article](https://opensource.googleblog.com/2022/12/open-sourcing-attention-center-model.html) to find more information about this model. The attention-center model takes an RGB image as input and return a 2D point as result, which is the predicted center of human attention on the image.


## Notebook Contents

The tutorial consists of the following steps:

* Downloading the model
* Loading the model and make inference with OpenVINO API
* Run Live Attention Center Detection

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).

### See Also

* [OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
* [Model Optimizer]( https://docs.openvino.ai/nightly/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)