# Live Action Recognition with OpenVINO™

<p align="center" width="100%">
    <img width="80%" src="https://user-images.githubusercontent.com/10940214/151552326-642d6e49-f5a0-4fc1-bf14-ae3f457e1fec.gif"> 
</p>

Human action recognition finds actions over time in a video. The list of actions in this notebook is extensive (400 in total) and covers *Person Actions*, (for example, drawing, drinking, laughing), *Person-Person Actions* (for example, hugging, shaking hands), and *Person-Object Actions* (for example, opening present, mowing the lawn, playing "instrument"). You could find several parent-child groupings on the list of labels, such as braiding hair and brushing hair, salsa dancing, robot dancing, or playing violin and playing guitar. For more information about the labels and the dataset, see the ["The Kinetics Human Action Video Dataset"](https://arxiv.org/pdf/1705.06950.pdf) research paper.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F403-action-recognition-webcam%2F403-action-recognition-webcam.ipynb)<br> *Binder is a free service where the webcam will not work, and performance on the video will not be good. For the best performance, install the notebooks locally.*

## Notebook Contents

This notebook demonstrates live human action recognition with OpenVINO, using the [Action Recognition Models](https://docs.openvino.ai/latest/omz_models_group_intel.html#action-recognition-models) from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo), specifically the Encoder and Decoder from [action-recognition-0001](https://docs.openvino.ai/latest/omz_models_model_action_recognition_0001.html). Both models create a sequence to sequence (`"seq2seq"`)<sup id="a1">[1](#f1)</sup> system to identify the  human activities for [Kinetics-400 dataset](https://deepmind.com/research/open-source/kinetics). The models use the Video Transformer approach with ResNet34 encoder<sup id="a2">[2](#f2)</sup>. The notebook shows how to create the following pipeline:

<p align="center" width="100%">
    <img width="80%" src="https://user-images.githubusercontent.com/10940214/148401661-477aebcd-f2d0-4771-b107-4b37f94d0b1e.jpeg"> 
</p>

Final part of this notebook shows live inference results from a webcam. Additionaly, you can also upload a video file.

> **NOTE**: To use the webcam, you must run this Jupyter notebook on a computer with a webcam. If you run on a server, the webcam will not work. However, you can still do inference on a video in the final step.

<b id="f1">1</b> seq2seq: Deep learning models that take a sequence of items to the input and output. In this case, input: video frames, output: actions sequence. This `"seq2seq"` is composed of an encoder and a decoder. The encoder captures `"context"` of the inputs to be analyzed by the decoder and finally gets the human action and confidence.[↩](#a1)

<b id="f2">2</b>  [Video Transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)#:~:text=A%20transformer%20is%20a%20deep,in%20computer%20vision%20(CV).), and [ResNet34](https://www.kaggle.com/pytorch/resnet34). [↩](#a2)

For more information about the pre-trained models, refer to the [Intel models documentation](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel) and [public](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public). All are included in the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).

### See Also

* [OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
* [Model Optimizer](https://docs.openvino.ai/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Action Recognition Demo (OpenVINO - No notebooks)](https://docs.openvino.ai/latest/omz_demos_action_recognition_demo_python.html)
