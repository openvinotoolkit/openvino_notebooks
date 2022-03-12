# Vehicle Detection And Recognition with OpenVINO

<div  align='center' ><img src="data/vehicle-result.png" alt="drawing"/></div>

Thousands of vehicles are running in road.We can detect a single vehicle from a raw image and recognize attributes of vehicles.In this notebook,we use detection model and recognition model to realize the target.


## Notebook Contents

In this notebook, we will use both detection model and classification model with OpenVINO.We use [Object Detection Models](https://docs.openvino.ai/2020.2/usergroup3.html) and [Object Recognition Models](https://docs.openvino.ai/2020.2/usergroup4.html) from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo).Detection model is used to detect vehicle position.Besides, we crop single vehicle and infer with classification model to recognize attributes of single vehicle.The pipline is here： 
<div  align='center' ><img src="data/vehicle-inference-flow.png" alt="drawing" width="1000"/></div>

For more information about the pre-trained models, refer to the [Intel](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel) and [public](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public) models documentation. All included in the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.