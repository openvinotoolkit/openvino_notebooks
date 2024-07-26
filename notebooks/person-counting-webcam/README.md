# Person Counting System using YOLOv8 and OpenVINO

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/person-counting-webcam/person-counting.ipynb)

In this project, we utilized the YOLOv8 Object Counting class to develop a real-time person counting system using the YOLOv8 object detection model and tracking, optimized for Intel's OpenVINO toolkit to enhance inferencing speed. This system effectively monitors the number of individuals entering and exiting a room, leveraging the optimized YOLOv8 model for accurate person detection under varied conditions.

By utilizing the OpenVINO runtime on Intel hardware, the system achieves significant improvements in processing speed, making it ideal for applications requiring real-time data, such as occupancy management and traffic flow control in public spaces and commercial settings.

References:

- YOLOv8 Object counting documentation: <a href="https://docs.ultralytics.com/guides/object-counting/" target="_blank">https://docs.ultralytics.com/guides/object-counting/</a>
- OpenVINO Jupyter Notebooks: <a href="https://github.com/openvinotoolkit/openvino_notebooks/" target="_blank">https://github.com/openvinotoolkit/openvino_notebooks/</a>

<div align="center"><img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/e0525f8a-4578-4c56-a0a5-ce68e30d2d45" width=900/></div>


## Performance

In this clip, you can see the difference (Inference time and FPS) between running YOLOv8 natively with PyTorch vs optimized with OpenVINO.

<div align="center"><img src="https://github.com/antoniomtz/people-counting-yolov8-openvino/raw/main/optimized.gif" width=900/></div>

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/person-counting-webcam/README.md" />
