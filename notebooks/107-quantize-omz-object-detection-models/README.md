# Quantize Open Model Zoo object detection models
Quantizing a model accelerates a trained model by reducing the precision necessary for its calculations.  Acceleration comes from lower-precision calculations being faster as well as less memory needed and less data to transfer since the data type itself is smaller along with the model weights data.  Though lower-precision may reduce model accuracy, typically a model using 32-bit floating-point precision (FP32) can be quantized to use lower-precision 8-bit integers (INT8) giving good results that are worth the trade off between accuracy and speed.  To see how quantization can accelerate models, see [INT8 vs FP32 Comparison on Select Networks and Platforms](https://docs.openvino.ai/latest/openvino_docs_performance_int8_vs_fp32.html#doxid-openvino-docs-performance-int8-vs-fp32) for some benchmarking results.

[Intel Distribution of OpenVINO toolkit](https://software.intel.com/openvino-toolkit) includes the [Post-Training Optimization Tool (POT)](https://docs.openvino.ai/latest/pot_README.html) to automate quantization.  For models available from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo), the [`omz_quantizer`](../104-model-tools/104-model-tools.ipynb) tool is available to automate running POT using its [DefaultQuantization](https://docs.openvino.ai/latest/pot_compression_algorithms_quantization_default_README.html#doxid-pot-compression-algorithms-quantization-default-r-e-a-d-m-e) 8-bit quantization algorithm to quantize models down to INT8 precision.

This Jupyter* Notebook will go step-by-step through the workflow of downloading either the [ssd_mobilenet_v1_coco](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd_mobilenet_v1_coco) or the [yolo-v4-tf](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v4-tf) model from the Open Model Zoo through quantization and then checking and benchmarking the results.  The workflow consists of following these steps:
1. Download and set up the the [Common Objects in Context (COCO)](https://cocodataset.org/) validation dataset to be used by omz_quantize
2. Download model from the Open Model Zoo
3. Convert model to FP32 IR files
4. Quantize FP32 model to create INT8 IR files
5. Run inference on original and quantized model
6. Check accuracy before and after quantization
7. Benchmark before and after quantization

While performing the steps above, the following [OpenVINO tools](../104-model-tools/104-model-tools.ipynb) will be used to download, convert, quantize, check accuracy, and benchmark the model:
- `omz_downloader` - Download model from the Open Model Zoo
- `omz_converter` - Convert an Open Model Zoo model
- `omz_quantizer` - Quantize an Open Model Zoo model
- `accuracy_check` - Check the accuracy of models using a validation dataset
- `benchmark_app` - Benchmark models

## About the models
This notebook is configurable to work with either of the two Open Model Zoo object detection models: ssd_mobilenet_v1_coco or yolo-v4-tf (the default).

### About the ssd_mobilenet_v1_coco model
The ssd_mobilenet_v1_coco model is a [Single-Shot multi-box Detection (SSD) network](https://arxiv.org/abs/1801.04381) that has been trained on the COCO dataset to perform object detection.  
The input to the converted model is a 300x300 BGR image.  The output of the model is an array of detection information for up to 100 objects giving the:
- image_id: image identifier of the image within the batch
- label: class identifier in the range of 1-91 for each class, plus one for background
- confidence: the prediction probability in the range of 0.0-1.0 for label
- (x_min, y_min): coordinates in normalized format (range 0.0-1.0) of the top-left of the bounding box
- (x_max, y_max): coordinates in normalized format (range 0.0-1.0) of the bottom-right of the bounding box

For details more details on the ssd_mobilenet_v1_coco model, see the Open Model Zoo [model](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd_mobilenet_v1_coco)  and the [paper](https://arxiv.org/abs/1801.04381).

### About the yolo-v4-tf model
The yolo-v4-tf model is a YOLO v4 real-time object detection model that was implemented in a Keras* framework and converted to a TensorFlow* framework.  The model was trained on the [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset with 80 classes.  The input to the converted model is a 608x608 BGR image.  The output of the model are arrays of detection boxes contained in the three output layers:
- StatefulPartitionedCall/model/conv2d_93/BiasAdd/Add: 76x76 
- StatefulPartitionedCall/model/conv2d_101/BiasAdd/Add: 38x38
- StatefulPartitionedCall/model/conv2d_109/BiasAdd/Add: 19x19

Each output layer contains an NxN array for different sized detection boxes within the original image.  Each detection box contains the following information:
- (x, y) - raw coordinates of box center, must apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get relative to the cell coordinates
- h, w - raw height and width of box, must apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get absolute height and width values
- box_score - confidence of detection box, must apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get confidence in 0.0-1.0 range
- class_no[80] - array of probability distribution over the 80 classes in logits format, must apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) and multiply by obtained confidence value to get confidence for each class

To reduce the results from the three output layers into distinct objects within the original image, the "intersection over union" (also known as the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)) algorithm is typically used to combine overlapping detection boxes with the same class into a single box containing the detected object.

For details more details on the yolo-v4-tf model, see the Open Model Zoo [model](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v4-tf), the paper ["YOLOv4: Optimal Speed and Accuracy of Object Detection"](https://arxiv.org/abs/2004.10934), and the [repository](https://github.com/david8862/keras-YOLOv3-model-set).