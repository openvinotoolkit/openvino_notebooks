# Frame interpolation with FILM and OpenVINO

[Frame interpolation](https://en.wikipedia.org/wiki/Motion_interpolation) is the process of synthesizing in-between images from a given set of images. The technique is often used for [temporal up-sampling](https://en.wikipedia.org/wiki/Frame_rate#Frame_rate_up-conversion) to increase the refresh rate of videos or to create slow motion effects. Nowadays, with digital cameras and smartphones, we often take several photos within a few seconds to capture the best picture. Interpolating between these “near-duplicate” photos can lead to engaging videos that reveal scene motion, often delivering an even more pleasing sense of the moment than the original photos.

![](https://github.com/googlestaging/frame-interpolation/raw/main/moment.gif)

In [\"FILM: Frame Interpolation for Large Motion\"](https://arxiv.org/pdf/2202.04901.pdf), published at ECCV 2022, a method to create high quality slow-motion videos from near-duplicate photos is presented. FILM is a new neural network architecture that achieves state-of-the-art results in large motion, while also handling smaller motions well.

The FILM model takes two images as input and outputs a middle image. At inference time, the model is recursively invoked to output in-between images. FILM has three components:
  1. Feature extractor that summarizes each input image with deep multi-scale (pyramid) features;
  2. Bi-directional motion estimator that computes pixel-wise motion (i.e., flows) at each pyramid level;
  3. Fusion module that outputs the final interpolated image.

FILM is trained on regular video frame triplets, with the middle frame serving as the ground-truth for supervision.

In this tutorial, we will use [TensorFlow Hub](https://tfhub.dev/) as a model source.

## Notebook contents
- Prerequisites
- Prepare images
- Load the model
- Infer the model
    - Single middle frame interpolation
    - Recursive frame generation
- Convert the model to OpenVINO IR
- Inference
    - Select inference device
    - Single middle frame interpolation
    - Recursive frame generation
- Interactive inference

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).