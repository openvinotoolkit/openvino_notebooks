# Vehicle Detection And Recognition with OpenVINO™

![result](https://user-images.githubusercontent.com/47499836/163544861-fa2ad64b-77df-4c16-b065-79183e8ed964.png)

This tutorial demonstrates how to use two pre-trained models from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo): [vehicle-detection-0200](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-detection-0200) for object detection and [vehicle-attributes-recognition-barrier-0039](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-attributes-recognition-barrier-0039) for image classification. Using these models, you will detect vehicles from raw images and recognize attributes of detected vehicles. 


## Notebook Contents

This notebook uses both a detection model and a classification model from Open Model Zoo. The number and location of vehicles in an image can be analyzed by using vehicle detection. Vehicle attribute recognition can assist in the statistics of vehicle characteristics in traffic analysis scenario. The detection model is used to detect vehicle position, which is then cropped to a single vehicle before it is sent to a classification model to recognize attributes of the vehicle. 

Overview of the pipeline: 
![flowchart](https://user-images.githubusercontent.com/47499836/157867076-9e997781-f9ef-45f6-9a51-b515bbf41048.png)

For more information about the pre-trained models, refer to the [Intel](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel) and [public](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public) models documentation from Open Model Zoo.

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
