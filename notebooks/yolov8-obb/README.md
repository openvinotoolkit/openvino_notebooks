# YOLOv8 Oriented Bounding Boxes Object Detection with OpenVINO™

The [YOLOv8](https://github.com/ultralytics/ultralytics) algorithm developed by [Ultralytics](https://ultralytics.com) is a cutting-edge, state-of-the-art (SOTA) model that is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection, image segmentation, image classification and keypoint detection tasks.

YOLO stands for “You Only Look Once”, it is a popular family of real-time object detection algorithms. The original YOLO object detector was first released in 2016. Since then, different versions and variants of YOLO have been proposed, each providing a significant increase in performance and efficiency. YOLOv8 builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. More details about its realization can be found in the [Ultralytics YOLOv8 Tasks documentation](https://docs.ultralytics.com/tasks/).


This tutorial explains how to convert and optimize the YOLOv8 PyTorch models with OpenVINO for [oriented bounding boxes object detection](https://docs.ultralytics.com/tasks/obb/) scenarios. Oriented object detection goes a step further than object detection and introduce an extra angle to locate objects more accurate in an image. The output of an oriented object detector is a set of rotated bounding boxes that exactly enclose the objects in the image, along with class labels and confidence scores for each box. Object detection is a good choice when you need to identify objects of interest in a scene, but don't need to know exactly where the object is or its exact shape.


This tutorial consists of the following steps:
- Prepare the PyTorch model.
- Download and prepare the dataset.
- Validate the original model.
- Convert the PyTorch model to OpenVINO IR.
- Validate the converted model.
- Prepare and run NNCF post-training optimization pipeline.
- Compare accuracy of the FP32 and quantized models.
- Compare performance of the FP32 and quantized models.


You can also try tutorials, which considered using object detection, instance segmentation and keypoint detection scenarios with OpenVINO Runtime:

- [yolov8-object-detection](../yolov8-object-detection/yolov8-object-detection.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov8-object-detection/yolov8-object-detection.ipynb)
- [yolov8-instance-segmentation](../yolov8-instance-segmentation/yolov8-instance-segmentation.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov8-instance-segmentation/yolov8-instance-segmentation.ipynb)
- [yolov8-keypoint-detection](../yolov8-keypoint-detection/yolov8-keypoint-detection.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov8-keypoint-detection/yolov8-keypoint-detection.ipynb)


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/yolov8-obb/README.md" />
