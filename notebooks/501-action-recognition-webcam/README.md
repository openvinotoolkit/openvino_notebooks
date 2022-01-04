# Human Action Recognition OpenVINO

![GIF ANIMATION HERE](./action_recognition.gif)

In this notebook you will find a simple way to detect human activities in a video coming from your camera or loaded video.

We will use a general-purpose action recognition (400 actions) model for Kinetics-400 dataset. `action-recognition-0001-encoder` + `action-recognition-0001-decoder`. But also you can use the notebook to run models as `driver-action-recognition-adas-0002-encoder` + `driver-action-recognition-adas-0002-decoder` for driver monitoring scenario. 

For more information about the pre-trained models, refer to the [Intel](../../../models/intel/index.md) and [public](../../../models/public/index.md) models documentation.

## Demo Output

The application uses OpenCV to display the real-time action recognition results and current inference performance (in FPS).

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
