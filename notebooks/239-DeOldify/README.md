# DeOldify Images with OpenVINO™

![Example](https://scontent.xx.fbcdn.net/v/t1.15752-9/336739817_1432069034257767_4406490209410590976_n.png?stp=dst-png_s640x640&_nc_cat=103&ccb=1-7&_nc_sid=aee45a&_nc_ohc=fzTPb_zqgaoAX-Mvi2Q&_nc_ad=z-m&_nc_cid=0&_nc_ht=scontent.xx&oh=03_AdSZtpWZTnmAkmS-dBhc9DgSRAEp2i8dwAihOtWVJU4oAw&oe=64479E9A)

The goal of the project is to use deep learning and computer vision techniques to restore colour information from a black and white images and videos thereby reviving them. This notebook demonstartes how to convert the pytorch model for DeOldify into a OpenVino IR format in order to make use of the OpenVINO optimisation tools and capabilities. There are now three models to choose from in DeOldify. Each of these has key strengths and weaknesses, and so have different use cases. Video is for video of course. But stable and artistic are both for images, and sometimes one will do images better than the other. Check out the [DeOldify repository](https://github.com/jantic/DeOldify) for more details.

## Notebook contents

This notebook demonstrates how to colorize images with OpenVINO using the DeOldify Artistic model, DeOldify stable and the DeOldify Video model.

## Installation Instructions

Follow the [installation guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/215-image-inpainting/README.md) if you haven't installed all of the necessary prerequisites.