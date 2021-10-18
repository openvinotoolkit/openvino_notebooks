# Object detection demo

![object detection](https://user-images.githubusercontent.com/4547501/137755014-6f174efb-f378-4fe2-9edd-cab7aab0df5f.jpg)

Object detection allows finding predefined objects on the image. Every returned object has features such: label, probability and box relative to image boundaries. 

List of predefined objects for this demo: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, be, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

## Notebook Contents

This notebook demonstrates object detection with OpenVINO using the [SSDLite MobileNetV2](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssdlite_mobilenet_v2) model from Open Model Zoo. The model is already pretrained on [COCO dataset](https://cocodataset.org/#home).

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
