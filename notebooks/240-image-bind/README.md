# Binding multimodal data using ImageBind and OpenVINO

Exploring the surrounding world, people get information using multiple senses, for example, seeing a busy street and hearing the sounds of car engines. ImageBind introduces an approach that brings machines one step closer to humans’ ability to learn simultaneously, holistically, and directly from many different forms of information. 
[ImageBind](https://github.com/facebookresearch/ImageBind) is the first AI model capable of binding data from six modalities at once, without the need for explicit supervision (the process of organizing and labeling raw data). By recognizing the relationships between these modalities — images and video, audio, text, depth, thermal, and inertial measurement units (IMUs) — this breakthrough helps advance AI by enabling machines to better analyze many different forms of information, together.

![ImageBind](https://user-images.githubusercontent.com/8495451/236859695-ffa13364-3e39-4d99-a8da-fbfab17f9a6b.gif)

In this tutorial, we consider how to convert and run ImageBind model using OpenVINO.


## Notebook Contents

The tutorial consists of following steps:

1. Download the pre-trained model.
2. Prepare input data examples.
3. Convert the model to OpenVINO Intermediate Representation format (IR).
4. Run model inference and analyze results.

We will use ImageBind model for zero-shot audio and image classification. The result of model work demonstrated on the image below

![image](https://user-images.githubusercontent.com/29454499/240364108-39868933-d221-41e6-9b2e-dac1b14ef32f.png)


## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).