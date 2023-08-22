# Quantization with accuracy control using NNCF

These tutorials demonstrate how to apply 8-bit quantization with accuracy control to the speech recognition and YOLOv8 models::
* `122-speech-recognition-quantization-wav2vec2.ipynb` demonstrates how to apply post-training `INT8` quantization with accuracy control on a fine-tuned [Wav2Vec2-Base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) [PyTorch](https://pytorch.org/) model, trained on the [LibriSpeech ASR corpus](https://www.openslr.org/12).
* `122-yolov8-quantization-with-accuracy-control.ipynb` demonstrates how to apply post-training `INT8` quantization on the [YOLOv8](https://github.com/ultralytics/) PyTorch model

The code of the tutorials is designed to be extendable to the same model types trained on custom datasets.

The advanced quantization flow allows to apply 8-bit quantization to the model with control of accuracy metric. This is achieved by keeping the most impactful operations within the model in the original precision. The flow is based on the [Quantizing with Accuracy Control](https://docs.openvino.ai/2023.0/quantization_w_accuracy_control.html) and has the following specifics:

- Besides the calibration dataset, a validation dataset is required to compute the accuracy metric. Both datasets can refer to the same data in the simplest case.
- Validation function, used to compute accuracy metric is required. It can be a function that is already available in the source framework or a custom function.
- Since accuracy validation is run several times during the quantization process, quantization with accuracy control can take more time than the Basic 8-bit quantization flow.
- The resulted model can provide smaller performance improvement than the Basic 8-bit quantization flow because some of the operations are kept in the original precision.

> **NOTE**: Currently, 8-bit quantization with accuracy control in NNCF is available only for models in OpenVINO representation.

> **NOTE**: Quantization with accuracy control can take a long time and requires a lot of RAM. See details in each of the examples.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
