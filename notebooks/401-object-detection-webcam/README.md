# Live Object Detection with OpenVINO

![object detection](https://user-images.githubusercontent.com/4547501/141471665-82b28c86-cf64-4bfe-98b3-c314658f2d96.gif)

Object detection finds predefined objects in an image or video. Each returned object includes features such as label, probability and bounding box coordinates relative to image boundaries. 

List of predefined objects available in this demo: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, be, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

## Notebook Contents

This notebook demonstrates object detection with OpenVINO using the [SSDLite MobileNetV2](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssdlite_mobilenet_v2) model from Open Model Zoo. The model was trained on the [COCO dataset](https://cocodataset.org/#home).

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
