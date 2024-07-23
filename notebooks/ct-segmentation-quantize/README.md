# Quantize a Segmentation Model and Show Live Inference

<p align="center">
    <img src="https://user-images.githubusercontent.com/77325899/154279555-aaa47111-c976-4e77-8d23-aac96f45872f.gif"/>
</p>

## Notebook Contents

This folder contains notebook that show how to quantize and show live inference on a [MONAI](https://monai.io/) segmentation model with OpenVINO,

NNCF performs quantization within the PyTorch framework. There is a pre-trained model and a subset of the dataset provided for the quantization notebook,
so it is not required to run the data preparation and training notebooks before running the quantization tutorial.

This quantization tutorial consists of the following steps:

* Use model conversion Python API to convert the model to OpenVINO IR. For more information about model conversion Python API, see this [page](https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html).
* Quantizing the model with NNCF with the [Post-training Quantization with NNCF Tool](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html) API in OpenVINO.
* Evaluating the F1 score metric of the original model and the quantized model.
* Benchmarking performance of the original model and the quantized model.
* Showing live inference with async API and MULTI plugin in OpenVINO.

You will also see real-time segmentation of kidney CT scans running on a CPU, iGPU, or combining both devices for higher
throughput. The processed frames are 3D scans that are shown as individual slices. The visualization slides through the slices with detected kidneys
overlayed in red. A pre-trained and quantized model is provided, so running the previous notebooks (1-3) in the series is not required.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/ct-segmentation-quantize/README.md" />
