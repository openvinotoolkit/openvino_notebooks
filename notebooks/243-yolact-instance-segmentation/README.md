
# Instance Segmentation with OpenVINO™ and ONNX using Yolact Model 

This notebook demonstrates how to convert and use [Yolact](https://github.com/dbolya/yolact) PyTorch model 
with OpenVINO.



![Inference example](https://raw.githubusercontent.com/Abdullah-Elkasaby/yolact-isntance-segmentation-openvino/main/coco_bike_yolact.png)


Instance segmentation is a crucial computer vision task that aims to detect and precisely delineate individual objects within an image, providing pixel-level segmentation masks for each instance. Unlike object detection, instance segmentation enables fine-grained localization and boundary understanding, opening up diverse applications in autonomous driving, robotics, and medical imaging, among others.


![Yolact diagram](https://raw.githubusercontent.com/Abdullah-Elkasaby/yolact-isntance-segmentation-openvino/main/model_diagram.png)
> Credits for this image go to [original authors of Yolact](https://arxiv.org/abs/1904.02689).


Key features and more details can be found in the original model:
[repository](https://github.com/dbolya/yolact).

## Contents


This tutorial demonstrates step-by-step instructions on how to run YOLACT with OpenVINO. 


* Download the pretrained weights.
* Prepare the PyTorch model.
* Validate the original model.
* Convert the PyTorch model to ONNX.
* Convert the ONNX model to OpenVINO IR.
* Validate the converted model.
* Run the model 

