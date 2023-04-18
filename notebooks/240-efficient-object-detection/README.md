# Efficient object detection with OpenVINO

![Result](https://user-images.githubusercontent.com/71766106/226086430-a7e3cdc4-1f99-4c46-89f9-60dcbadea44a.png)

There are a lot of object detection models with great precision and performance, but when it comes to applications, the model's scalability and efficiency also play a crucial role. EfficientDet is one of the few object detection models that is considered efficient and scalable while achieving state-of-the-art accuracy.

EfficientDet achieves accuracy of 55.1mAP on COCO test-dev, while being 4x - 9x smaller and using 13x - 42x fewer FLOPs than other detectors. EfficientDet models also run 2x - 4x faster on GPU, and 5x - 11x faster on CPU than other detectors.

Research Paper : [EfficientDet: Scalable and Efficient Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.pdf)\
Official GitHub : [EfficientDet](https://github.com/google/automl/tree/master/efficientdet)

## Notebook Contents
This notebook demonstrates OpenVINO implementation of the EfficientDet object detection model.
Notebook contains the following steps:
* Downloading and using EfficientDet-d0 with TensorFlow-HUB.
* Converting the EfficientDet tf-model to OpenVINO-IR (Intermediate representation) file. Then, it's inference with OpenVINO runtime.
* Validating accuracy on COCO dataset.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
