# Quantize the Open Model Zoo resnet-50-tf model
Quantizing a model accelerates a trained model by reducing the precision necessary for its calculations.  Acceleration comes from lower-precision calculations being faster as well as less memory needed and less data to transfer since the data type itself is smaller along with the model weights data.  Though lower-precision may reduce model accuracy, typically a model using 32-bit floating-point precision (FP32) can be quantized to use lower-precision 8-bit integers (INT8) giving good results that are worth the trade off between accuracy and speed.  To see how quantization can accelerate models, see [INT8 vs FP32 Comparison on Select Networks and Platforms](https://docs.openvino.ai/latest/openvino_docs_performance_int8_vs_fp32.html#doxid-openvino-docs-performance-int8-vs-fp32) for some benchmarking results.

[Intel Distribution of OpenVINO toolkit](https://software.intel.com/openvino-toolkit) includes the [Post-Training Optimization Tool (POT)](https://docs.openvino.ai/latest/pot_README.html) to automate quantization.  For models available from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo), the [`omz_quantizer`](https://pypi.org/project/openvino-dev/) tool is available to automate running POT using its [DefaultQuantization](https://docs.openvino.ai/latest/pot_compression_algorithms_quantization_default_README.html#doxid-pot-compression-algorithms-quantization-default-r-e-a-d-m-e) 8-bit quantization algorithm to quantize models down to INT8 precision.

This Jupyter* Notebook will go step-by-step through the workflow of downloading the [resnet-50-tf](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf) model from the Open Model Zoo through quantization and then checking and benchmarking the results.  The workflow consists of following the steps:
1. Download and set up the the [Imagenette](https://github.com/fastai/imagenette) (subset of [ImageNet](http://www.image-net.org/)) validation dataset to be used by omz_quantize
2. Download model
3. Convert model to FP32 IR files
4. Quantize FP32 model to create INT8 IR files
5. Run inference on original and quantized model
6. Check accuracy before and after quantization
7. Benchmark before and after quantization

While performing the steps above, the following [OpenVINO tools](https://pypi.org/project/openvino-dev/) will be used to download, convert, quantize, check accuracy, and benchmark the model:
- `omz_downloader` - Download model from the Open Model Zoo
- `omz_converter` - Convert an Open Model Zoo model
- `omz_quantizer` - Quantize an Open Model Zoo model
- `accuracy_check` - Check the accuracy of models using a validation dataset
- `benchmark_app` - Benchmark models

# About the model
This notebook uses the 'resnet-50-tf' which is a TensorFlow* implementation of ResNet-50, an image classification model that has been trained on the ImageNet dataset. The input to the converted model is a 224x224 BGR image.  The output of the model is 1001 prediction probabilities in the range of 0.0-1.0 for each of the 1000 classes, plus one for background.

For details more details on the model, see the Open Model Zoo [model](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf), the [paper](https://arxiv.org/abs/1512.03385) and the [repository](https://github.com/tensorflow/models/tree/v2.2.0/official/r1/resnet).