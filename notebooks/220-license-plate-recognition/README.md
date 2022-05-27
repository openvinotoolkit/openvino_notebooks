# License Plate Recognition with OpenVINO

License plate recognition model helps you get the chinese license plate number precisely in no time. The input of the color license plate image can be any size. It will be resized and augmented before put into the model. After matching the result to correct character, we can get the license plate number. In the notebook we show how to create the following pipeline:

![text](https://user-images.githubusercontent.com/15709723/162659593-3f620d7a-44d2-4f49-a558-94c35a244a8e.png)

And the result is:

![text](https://user-images.githubusercontent.com/70456146/162759539-4a0a996f-dabe-40ea-98d6-85b4dce8511d.png)

## Notebook Contents

This notebook demonstrates license plate recognition with OpenVINO. We use the [License Plate Recognition Model](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/license-plate-recognition-barrier-0001) from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/). This model uses a small-footprint network trained end-to-end to recognize Chinese license plates in traffic.

Here is the [Jupyter notebook](220-license-plate-recognition.ipynb)

## Installation Instructions

If you have not done so already, please follow the [Installation Guide](../../README.md) to install all required dependencies.
