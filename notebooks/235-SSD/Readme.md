# GSOC 2023 Prequisite Task 
1. Converted MobileNetV2 model from Tensorflow to OpenVINO IR format.
2. Link to [mscoco_label_map.pbtxt file](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt) 
3. Label Map was not available anywhere in the model, my apologies for having to download it separately.
4. This notebook uses an image for inference that is not part of the COCO the dataset.
5. I have avoided using TensorFlow object detection API and protobuf for this task, as it is not a part of current requirements.txt
