# DeOldify Images with OpenVINO™

![Example](https://media.githubusercontent.com/media/dana-kelley/DeOldify/master/result_images/FlyingOverEdinburgh.jpg)

The goal of the project is to use deep learning and computer vision techniques to restore colour information from a black and white images and videos thereby reviving them. This notebook demonstartes how to convert the pytorch model for DeOldify into a OpenVino IR format in order to make use of the OpenVINO optimisation tools and capabilities. There are now three models to choose from in DeOldify. Each of these has key strengths and weaknesses, and so have different use cases. Video is for video of course. But stable and artistic are both for images, and sometimes one will do images better than the other. Check out the [DeOldify repository](https://github.com/jantic/DeOldify) for more details.

## Notebook contents

This notebook demonstrates how to colorize images with OpenVINO using the DeOldify Artistic model, DeOldify stable and the DeOldify Video model.

## Installation Instructions

Follow the [installation guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/215-image-inpainting/README.md) if you haven't installed all of the necessary prerequisites.
