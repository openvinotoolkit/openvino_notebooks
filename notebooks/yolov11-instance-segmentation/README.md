# Convert and Optimize YOLOv11 instance segmentation model with OpenVINO™

<p align="center">
    <img src="https://cdn.prod.website-files.com/6479eab6eb2ed5e597810e9e/67ed55c84b7add409d70b313_6729fecc79bc58a68880f373_6729fd101d9e509019292e2a_Segmentation_Fig%2525201.png"/>
</p>
[**image source*](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-instance-segmentation)

The [YOLOv11](https://github.com/ultralytics/ultralytics) algorithm developed by [Ultralytics](https://ultralytics.com) is a cutting-edge, state-of-the-art (SOTA) model that is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection, image segmentation, image classification and keypoint detection tasks.

YOLO stands for “You Only Look Once”, it is a popular family of real-time object detection algorithms. The original YOLO object detector was first released in 2016. Since then, different versions and variants of YOLO have been proposed, each providing a significant increase in performance and efficiency. YOLOv11 builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. More details about its realization can be found in the [Ultralytics YOLOv11 Tasks documentation](https://docs.ultralytics.com/tasks/).

This tutorial explains how to convert and optimize the [YOLOv11 PyTorch models](https://docs.ultralytics.com/models/yolo11/) with OpenVINO for [instance segmentation scenarios](https://docs.ultralytics.com/tasks/segment/). Instance segmentation goes a step further than object detection and involves identifying individual objects in an image and segmenting them from the rest of the image. Instance segmentation as an object detection are often used as key components in computer vision systems.


This tutorial consists of the following steps:
- Prepare the PyTorch model.
- Convert the PyTorch model to OpenVINO IR.
- Validate the converted model.
- Prepare and run NNCF post-training optimization pipeline.
- Compare performance of the FP32 and quantized models.


You can also try tutorials, which considered using object detection and keypoint detection scenarios with OpenVINO Runtime:

- [yolov11-object-detection](../yolov11-object-detection/yolov11-object-detection.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov11-object-detection/yolov11-object-detection.ipynb)
- [yolov11-keypoint-detection](../yolov11-keypoint-detection/yolov11-keypoint-detection.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov11-keypoint-detection/yolov11-keypoint-detection.ipynb)


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/yolov11-instance-segmentation/README.md" />
