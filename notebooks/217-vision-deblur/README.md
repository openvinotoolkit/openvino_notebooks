# Deblur Images with DeblurGAN-v2 and OpenVINO™
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ThanosM97/openvino_notebooks/217-vision-deblur?labpath=notebooks%2F217-vision-deblur%2F217-vision-deblur.ipynb)

Deblurring is the task of removing motion blurs that usually occur in photos shot with hand-held cameras, when there are moving objects in the scene. Blurs not only reduce the human perception about the quality of the image, but also complicate computer vision analyses.

![result](https://user-images.githubusercontent.com/41332813/158425051-3d4d442c-7eca-4f5c-97c8-de27e0ea8093.png)

## Notebook Contents

This tutorial demonstrates Single Image Motion Deblurring with DeblurGAN-v2 in OpenVINO, by first converting the [VITA-Group/DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2) model to OpenVINO Intermediate Representation (OpenVINO IR) format. For more information about the model, see the [deblurgan-v2 model documentation](https://docs.openvino.ai/latest/omz_models_model_deblurgan_v2.html).

For more information, refer to the following research paper: 

Kupyn, O., Martyniuk, T., Wu, J., & Wang, Z. (2019). [Deblurgan-v2: Deblurring (orders-of-magnitude) faster and better.](https://openaccess.thecvf.com/content_ICCV_2019/html/Kupyn_DeblurGAN-v2_Deblurring_Orders-of-Magnitude_Faster_and_Better_ICCV_2019_paper.html) In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 8878-8887).

### The Data

The image above is a sample of the test subset of the [GOPRO](https://seungjunnah.github.io/Datasets/gopro) dataset. This dataset is one of the three that were used to train DeblurGAN-v2, and contains pairs of blurred and sharp images.

The authors release the dataset under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](../../README.md).
