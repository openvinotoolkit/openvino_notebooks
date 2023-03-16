# EfficientDet Optimization with OpenVINO
<a href='https://github.com/google/automl/tree/master/efficientdet'>EfficientDets</a> are a family of object detection models, which achieve state-of-the-art 55.1mAP on COCO test-dev, yet being 4x - 9x smaller and using 13x - 42x fewer FLOPs than previous detectors. Our models also run 2x - 4x faster on GPU, and 5x - 11x faster on CPU than other detectors

## Notebook Content

* Downloading and exporting the efficientDet-d0 model from TensorFlow-HUB
  * Validating inference of tf-model
* Converting tf-model to OpenVINO-IR(Intermediate representation) file {FP16}
  * Validating inference of OpenVINO-model
  * Evaluating performance by AP( average Precision)

[Will be included later]
* Post-training optimization with OpenVINO NNCF (Neural Network Compression Framework)
  * Quantization of FP16/32 model to INT8 model
  * Validating Inference of INT8 model
  * Evaluating performance of INT8 model by AP

<img src='https://i.ibb.co/FB8bLxd/download.png'>

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
