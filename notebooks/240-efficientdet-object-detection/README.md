# EfficientDet Optimization with OpenVINO

![Result](https://user-images.githubusercontent.com/71766106/226086430-a7e3cdc4-1f99-4c46-89f9-60dcbadea44a.png)

Research Paper : [EfficientDet: Scalable and Efficient Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.pdf)
[EfficientDets](https://github.com/google/automl/tree/master/efficientdet) are a family of object detection models, which achieve state-of-the-art 55.1mAP on COCO test-dev, yet being 4x - 9x smaller and using 13x - 42x fewer FLOPs than previous detectors. EfficientDets also run 2x - 4x faster on GPU, and 5x - 11x faster on CPU than other detectors

## Notebook Content

* Downloading and exporting the efficientDet-d0 from TensorFlow-HUB.
  * Validating inference of tf-model.
* Converting tf-model to OpenVINO-IR(Intermediate representation) file.
  * Validating inference of OpenVINO-model.
  * Evaluating performance by AP( average Precision).

[Will be included later]
* Post-training optimization with OpenVINO NNCF (Neural Network Compression Framework).
  * Quantization of FP16/32 model to INT8 model.
  * Validating Inference of INT8 model.
  * Evaluating performance of INT8 model by AP.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
