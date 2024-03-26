# Video Recognition using SlowFast and OpenVINO™
The goal of this notebook is to demonstrate how to convert and run inference on a SlowFast model using OpenVINO. Specifically, a pretrained ResNet-50 based SlowFast Network downloaded from the `pytorchvideo` repository is converted and inferred.

The SlowFast model is a video recognition network that achieved strong performance for both action classification and detection in video and reported state-of-the-art accuracy on major video recognition benchmarks such as Kinetics-400 and AVA. The intuition behind the SlowFast model is that some actions in videos occur slowly over a longer period of time, while others happen quickly and require a higher temporal resolution to be accurately detected. For example, waving hands do not change their identity as “hands” over the span of the waving action, and a person is always in the “person” category even though he/she can transit from walking to running. So recognition of this manner can be refreshed relatively slowly. On the other hand, the motion being performed needs faster refresh to be effectively modelled. 

<div align=center>
<img src="https://github.com/facebookresearch/SlowFast/raw/main/demo/ava_demo.gif" width="800"/>
</div>

To achieve this, the SlowFast model uses a slow pathway operating at a low frame-rate to analyze the static content and a fast pathway operating at a high frame-rate to capture dynamic content. Although two-stream network architectures have been proposed before, the key concept of different temporal speeds allows it to effectively capture both fast and slow motion information in video sequences, making it particularly well-suited to tasks that require a temporal and spatial understanding of the data.

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143044111-94676f64-7ba8-4081-9011-f8054bed7030.png" width="800"/>
</div>


More details about the network can be found in the original paper: [SlowFast Networks for Video Recognition](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html), and [repository](https://github.com/facebookresearch/SlowFast).

## Notebook Contents

This tutorial consists of the following steps

- Preparing the PyTorch model
- Download and prepare data
- Check inference with the PyTorch model
- Convert Model to OpenVINO Intermediate Representation
- Verify inference with the converted model

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md). 
Additional dependencies will be installed in the notebook.