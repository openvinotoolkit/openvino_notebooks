# Human Action Recognition OpenVINO

![GIF ANIMATION HERE](./action_recognition.gif)

In this notebook, you will find a simple way to detect human activities in a video coming from your camera or loaded video.

We will use a general-purpose action recognition (400 actions) model for Kinetics-400 dataset. `action-recognition-0001-encoder` + `action-recognition-0001-decoder`. But also you can use the notebook to run models as `driver-action-recognition-adas-0002-encoder` + `driver-action-recognition-adas-0002-decoder` for driver monitoring scenario. 

For more information about the pre-trained models, refer to the [Intel](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel) and [public](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public) models documentation. All included in the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)

## Demo Output

In the bottom part of the notebook, you can find the result of running `"action-recognition-model"` using your webcam and a loaded video.

## See Also

* [OpenVINO notebooks](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Fix Link - Model Downloader](../../../tools/model_tools/README.md)
