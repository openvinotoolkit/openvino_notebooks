# Convert and Optimize YOLOv8 with OpenVINO™

<p align="center">
    <img src="https://user-images.githubusercontent.com/29454499/212105105-f61c8aab-c1ff-40af-a33f-d0ed1fccc72e.png"/>
</p>

The [YOLOv8](https://github.com/ultralytics/ultralytics) algorithm developed by [Ultralytics](https://ultralytics.com) is a cutting-edge, state-of-the-art (SOTA) model that is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection, image segmentation, image classification and keypoint detection tasks.

YOLO stands for “You Only Look Once”, it is a popular family of real-time object detection algorithms. The original YOLO object detector was first released in 2016. Since then, different versions and variants of YOLO have been proposed, each providing a significant increase in performance and efficiency. YOLOv8 builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. More details about its realization can be found in the [Ultralytics YOLOv8 Tasks documentation](https://docs.ultralytics.com/tasks/).


This tutorial explains how to convert and optimize the YOLOv8 PyTorch models with OpenVINO. These tutorials are considered object detection, instance segmentation and keypoint detection scenarios:

- [yolov8-object-detection](./yolov8-object-detection.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov8-optimization/yolov8-object-detection.ipynb)
- [yolov8-instance-segmentation](./yolov8-instance-segmentation.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov8-optimization/yolov8-instance-segmentation.ipynb)
- [yolov8-keypoint-detection](./yolov8-keypoint-detection.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov8-optimization/yolov8-keypoint-detection.ipynb)
- [yolov8-obb](./yolov8-obb.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov8-optimization/yolov8-obb.ipynb)


Each case tutorial consists of the following steps::
- Prepare the PyTorch model.
- Download and prepare the dataset.
- Validate the original model.
- Convert the PyTorch model to OpenVINO IR.
- Validate the converted model.
- Prepare and run NNCF post-training optimization pipeline.
- Compare accuracy of the FP32 and quantized models.
- Compare performance of the FP32 and quantized models.


## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
