English | [简体中文](README_cn.md)

<h1>📚 OpenVINO™ Notebooks</h1>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/LICENSE)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/treon_precommit.yml/badge.svg?event=push)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/treon_precommit.yml?query=event%3Apush)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml/badge.svg?event=push)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml?query=event%3Apush)

A collection of ready-to-run Jupyter notebooks for learning and experimenting with the OpenVINO™ Toolkit. The notebooks provide an introduction to OpenVINO basics and teach developers how to leverage our API for optimized deep learning inference.

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

## 🚀 AI Trends - Notebooks
Check out the latest notebooks that show how to optimize and deploy popular models on Intel CPU and GPU. 

| **Notebook** | **Description** | **Preview** | **Complementary Materials** |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [YOLOv8 - Optimization](notebooks/230-yolov8-optimization/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/230-yolov8-optimization/230-yolov8-optimization.ipynb) | Optimize YOLOv8 using NNCF PTQ API | <img src = "https://user-images.githubusercontent.com/29454499/212105105-f61c8aab-c1ff-40af-a33f-d0ed1fccc72e.png" width=300>  | [Blog - How to get YOLOv8 Over 1000 fps with Intel GPUs?](https://medium.com/openvino-toolkit/how-to-get-yolov8-over-1000-fps-with-intel-gpus-9b0eeee879) |
| [SAM - Segment Anything Model](notebooks/237-segment-anything/)<br>| Prompt based object segmentation mask generation using Segment Anything and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/231468849-1cd11e68-21e2-44ed-8088-b792ef50c32d.png width=300> |  [Blog - SAM: Segment Anything Model — Versatile by itself and Faster by OpenVINO](https://medium.com/@paularamos_5416/sam-segment-anything-model-versatile-by-itself-and-faster-by-openvino-50175f06cd24)  |
| [ControlNet - Stable-Diffusion](notebooks/235-controlnet-stable-diffusion/)<br>| A Text-to-Image Generation with ControlNet Conditioning and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/224541412-9d13443e-0e42-43f2-8210-aa31820c5b44.png width=300> | [Blog - Control your Stable Diffusion Model with ControlNet and OpenVINO](https://medium.com/@paularamos_5416/control-your-stable-diffusion-model-with-controlnet-and-openvino-f2aa7e6b1ebd)  |
| [Stable Diffusion v2](notebooks/236-stable-diffusion-v2/)<br>| Text-to-Image Generation and Infinite Zoom with Stable Diffusion v2 and OpenVINO™ |  <img src=https://user-images.githubusercontent.com/1720147/229233760-79c9425e-5691-4114-ad13-7e33f9327b52.gif width=300>  | [Blog - How to run Stable Diffusion on Intel GPUs with OpenVINO](https://medium.com/openvino-toolkit/how-to-run-stable-diffusion-on-intel-gpus-with-openvino-840714f122b4)  |
| [Whisper - Subtitles generation](notebooks/227-whisper-subtitles-generation/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/227-whisper-subtitles-generation/227-whisper-subtitles-generation.ipynb) | Generate subtitles for video with OpenAI Whisper and OpenVINO | <img src=https://user-images.githubusercontent.com/29454499/204548693-1304ef33-c790-490d-8a8b-d5766acb6254.png  width=300>  |   |
| [CLIP - zero-shot-image-classification](notebooks/228-clip-zero-shot-image-classification)<br> | Perform Zero-shot Image Classification with CLIP and OpenVINO | <img src=https://user-images.githubusercontent.com/29454499/207795060-437b42f9-e801-4332-a91f-cc26471e5ba2.png width=300>  | [Blog - Generative AI and Explainable AI with OpenVINO ](https://medium.com/@paularamos_5416/generative-ai-and-explainable-ai-with-openvino-2b5f8e4e720b#:~:text=pix2pix%2Dimage%2Dediting-,Explainable%20AI%20with%20OpenVINO,-Explainable%20AI%20is)  |
| [BLIP - Visual-language-processing](notebooks/233-blip-visual-language-processing/)<br>| Visual Question Answering and Image Captioning using BLIP and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/221933762-4ff32ecb-5e5d-4484-80e1-e9396cb3c511.png width=300>  | [Blog - Multimodality with OpenVINO — BLIP](https://medium.com/@paularamos_5416/multimodality-with-openvino-blip-b20bd3a2c87)  |
| [Instruct pix2pix - Image-editing](notebooks/231-instruct-pix2pix-image-editing/)<br>| Image editing with InstructPix2Pix | <img src=https://user-images.githubusercontent.com/29454499/219943222-d46a2e2d-d348-4259-8431-37cf14727eda.png width=300>  | [Blog - Generative AI and Explainable AI with OpenVINO](https://medium.com/@paularamos_5416/generative-ai-and-explainable-ai-with-openvino-2b5f8e4e720b#:~:text=2.-,InstructPix2Pix,-Pix2Pix%20is%20a)  |
| [DeepFloyd IF - Text-to-Image generation](notebooks/238-deeployd-if/)<br>| Text-to-Image Generation with DeepFloyd IF and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/241643886-dfcf3c48-8d50-4730-ae28-a21595d9504f.png width=300> |   |
| [ImageBind](notebooks/239-image-bind/)<br>| Binding multimodal data using ImageBind and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/240364108-39868933-d221-41e6-9b2e-dac1b14ef32f.png width=300> |   |
| [Dolly v2](notebooks/240-dolly-2-instruction-following/)<br>| Instruction following using Databricks Dolly 2.0 and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/237291423-022f07d2-966b-4be2-9a1c-98f1cf0691c2.png width=300> | |
| [Stable Diffusion XL](notebooks/248-stable-diffusion-xl/)<br>| Image generation with Stable Diffusion XL and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/258651862-28b63016-c5ff-4263-9da8-73ca31100165.jpeg width=300> | |

## Table of Contents

- [🚀 AI Trends - Notebooks](#-ai-trends---notebooks)
- [Table of Contents](#table-of-contents)
- [📝 Installation Guide](#-installation-guide)
- [🚀 Getting Started](#-getting-started)
	- [💻 First steps](#-first-steps)
	- [⌚ Convert \& Optimize](#-convert--optimize)
	- [🎯 Model Demos](#-model-demos)
	- [🏃 Model Training](#-model-training)
	- [📺 Live Demos](#-live-demos)
- [⚙️ System Requirements](#️-system-requirements)
- [💻 Run the Notebooks](#-run-the-notebooks)
	- [To Launch a Single Notebook](#to-launch-a-single-notebook)
	- [To Launch all Notebooks](#to-launch-all-notebooks)
- [🧹 Cleaning Up](#-cleaning-up)
- [⚠️ Troubleshooting](#️-troubleshooting)
- [🧑‍💻 Contributors](#-contributors)
- [❓ FAQ](#-faq)

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

<div id='-installation-guide'/>

## 📝 Installation Guide

OpenVINO Notebooks require Python and Git. To get started, select the guide for your operating system or environment:

| [Windows](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) | [Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu) | [macOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS) | [Red Hat](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [CentOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [Azure ML](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML) | [Docker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Docker) | [Amazon SageMaker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/SageMaker) |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

<div id='-getting-started'/>

## 🚀 Getting Started

The Jupyter notebooks are categorized into four classes, select one related to your needs or give them all a try. Good Luck!

**NOTE: The main branch of this repository was updated to support the new OpenVINO 2023.0 release.** To upgrade to the new release version, please run `pip install --upgrade -r requirements.txt` in your `openvino_env` virtual environment. If you need to install for the first time, see the [Installation Guide](#-installation-guide) section below. If you wish to use the previous Long Term Support (LTS) version of OpenVINO check out the [2022.3 branch](https://github.com/openvinotoolkit/openvino_notebooks/tree/2022.3). 

If you need help, please start a GitHub [Discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions).  

<div id='-first-steps'/>

### 💻 First steps

Brief tutorials that demonstrate how to use OpenVINO's Python API for inference.

| [001-hello-world](notebooks/001-hello-world/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F001-hello-world%2F001-hello-world.ipynb) | [002-openvino-api](notebooks/002-openvino-api/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F002-openvino-api%2F002-openvino-api.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/002-openvino-api/002-openvino-api.ipynb) | [003-hello-segmentation](notebooks/003-hello-segmentation/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F003-hello-segmentation%2F003-hello-segmentation.ipynb) | [004-hello-detection](notebooks/004-hello-detection/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F004-hello-detection%2F004-hello-detection.ipynb) | 
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |  
| Classify an image with OpenVINO | Learn the OpenVINO Python API | Semantic segmentation with OpenVINO | Text detection with OpenVINO  | 
| <img src="https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg" width=140> | <img src="https://user-images.githubusercontent.com/15709723/127787560-d8ec4d92-b4a0-411f-84aa-007e90faba98.png" width=250> | <img src="https://user-images.githubusercontent.com/15709723/128290691-e2eb875c-775e-4f4d-a2f4-15134044b4bb.png" width=150> | <img src="https://user-images.githubusercontent.com/36741649/128489933-bf215a3f-06fa-4918-8833-cb0bf9fb1cc7.jpg" width=150>  | 

<div id='-convert--optimize'></div>

### ⌚ Convert & Optimize

Tutorials that explain how to optimize and quantize models with OpenVINO tools.

| Notebook | Description                                                                                                                 |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |:----------------------------------------------------------------------------------------------------------------------------|
| [101-tensorflow-classification-to-openvino](notebooks/101-tensorflow-classification-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F101-tensorflow-classification-to-openvino%2F101-tensorflow-classification-to-openvino.ipynb) | Convert TensorFlow models to OpenVINO IR                                                                                    |
| [102-pytorch-to-openvino](notebooks/102-pytorch-to-openvino/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/102-pytorch-to-openvino/102-pytorch-to-openvino.ipynb) | Convert PyTorch models to OpenVINO IR                                                                                |
| [103-paddle-to-openvino](notebooks/103-paddle-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F103-paddle-to-openvino%2F103-paddle-to-openvino-classification.ipynb) | Convert PaddlePaddle models to OpenVINO IR                                                                                  |
| [104-model-tools](notebooks/104-model-tools/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F104-model-tools%2F104-model-tools.ipynb) | Download, convert and benchmark models from Open Model Zoo                                                                  |
| [105-language-quantize-bert](notebooks/105-language-quantize-bert/) | Optimize and quantize a pre-trained BERT model                                                                              |
| [106-auto-device](notebooks/106-auto-device/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F106-auto-device%2F106-auto-device.ipynb) | Demonstrate how to use AUTO Device                                                                                          |
| [107-speech-recognition-quantization](notebooks/107-speech-recognition-quantization/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/107-speech-recognition-quantization/107-speech-recognition-quantization-data2vec.ipynb) | Quantize speech recognition models using NNCF PTQ API                                                                       |
| [108-gpu-device](notebooks/108-gpu-device/) | Working with GPUs in OpenVINO™                                                                                              |
| [109-performance-tricks](notebooks/109-performance-tricks/)| Performance tricks in OpenVINO™                                                                                             |
| [110-ct-segmentation-quantize](notebooks/110-ct-segmentation-quantize/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F110-ct-segmentation-quantize%2F110-ct-scan-live-inference.ipynb) | Quantize a kidney segmentation model and show live inference                                                                | 
| [111-yolov5-quantization-migration](notebooks/111-yolov5-quantization-migration)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/111-yolov5-quantization-migration/111-yolov5-quantization-migration.ipynb) | Migrate YOLOv5 POT API based quantization pipeline on Neural Network Compression Framework (NNCF)                           | <img src = "https://user-images.githubusercontent.com/44352144/177097174-cfe78939-e946-445e-9fce-d8897417ef8e.png"  width=225> |
| [112-pytorch-post-training-quantization-nncf](notebooks/112-pytorch-post-training-quantization-nncf/) | Use Neural Network Compression Framework (NNCF) to quantize PyTorch model in post-training mode (without model fine-tuning) | 
| [113-image-classification-quantization](notebooks/113-image-classification-quantization/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F113-image-classification-quantization%2F113-image-classification-quantization.ipynb) | Quantize Image Classification model                                                                                     |
| [115-async-api](notebooks/115-async-api/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F115-async-api%2F115-async-api.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/115-async-api/115-async-api.ipynb) | Use Asynchronous Execution to Improve Data Pipelining                                                                       | |
| [116-sparsity-optimization](notebooks/116-sparsity-optimization/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/116-sparsity-optimization/116-sparsity-optimization.ipynb) | Improve performance of sparse Transformer models                                                                            |
| [117-model-server](notebooks/117-model-server/)| Introduction to model serving with OpenVINO™ Model Server (OVMS)                                                            |
| [118-optimize-preprocessing](notebooks/118-optimize-preprocessing/)| Improve performance of image preprocessing step                                                                             |
| [119-tflite-to-openvino](notebooks/119-tflite-to-openvino/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/119-tflite-to-openvino/119-tflite-to-openvino.ipynb) | Convert TensorFlow Lite models to OpenVINO IR                                                                                                            |                                                                                                            |
| [120-tensorflow-object-detection-to-openvino](notebooks/120-tensorflow-object-detection-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F120-tensorflow-object-detection-to-openvino%2F120-tensorflow-object-detection-to-openvino.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/120-tensorflow-object-detection-to-openvino/120-tensorflow-object-detection-to-openvino.ipynb) | Convert TensorFlow Object Detection models to OpenVINO IR                                                                                                            |
| [121-convert-to-openvino](notebooks/121-convert-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F121-convert-to-openvino%2F121-convert-to-openvino.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/121-convert-to-openvino/121-convert-to-openvino.ipynb) | Learn OpenVINO model conversion API                                                                                                           |
<div id='-model-demos'></div>

### 🎯 Model Demos

Demos that demonstrate inference on a particular model.

| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [201-vision-monodepth](notebooks/201-vision-monodepth/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F201-vision-monodepth%2F201-vision-monodepth.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/201-vision-monodepth/201-vision-monodepth.ipynb) | Monocular depth estimation with images and video | <img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif" width=250> |
| [202-vision-superresolution-image](notebooks/202-vision-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-image.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/202-vision-superresolution/202-vision-superresolution-image.ipynb) | Upscale raw images with a super resolution model | <img src="https://user-images.githubusercontent.com/36741649/170005347-e4409f9e-ec34-416b-afdf-a9d8185929ca.jpg" width="70">→<img src="https://user-images.githubusercontent.com/36741649/170005347-e4409f9e-ec34-416b-afdf-a9d8185929ca.jpg" width="130"> |
| [202-vision-superresolution-video](notebooks/202-vision-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-video.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/202-vision-superresolution/202-vision-superresolution-video.ipynb) | Turn 360p into 1080p video using a super resolution model | <img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width=80>→<img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width="125"> |
| [203-meter-reader](notebooks/203-meter-reader/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F203-meter-reader%2F203-meter-reader.ipynb) | PaddlePaddle pre-trained models to read industrial meter's value | <img src="https://user-images.githubusercontent.com/91237924/166135627-194405b0-6c25-4fd8-9ad1-83fb3a00a081.jpg" width=225> |
| [204-segmenter-semantic-segmentation](notebooks/204-segmenter-semantic-segmentation/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/204-segmenter-semantic-segmentation/204-segmenter-semantic-segmentation.ipynb) | Semantic Segmentation with OpenVINO™ using Segmenter | <img src=https://user-images.githubusercontent.com/61357777/223854308-d1ac4a39-cc0c-4618-9e4f-d9d4d8b991e8.jpg width=225> |
| [205-vision-background-removal](notebooks/205-vision-background-removal/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F205-vision-background-removal%2F205-vision-background-removal.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/205-vision-background-removal/205-vision-background-removal.ipynb) | Remove and replace the background in an image using salient object detection | <img src="https://user-images.githubusercontent.com/15709723/125184237-f4b6cd00-e1d0-11eb-8e3b-d92c9a728372.png" width=455> |
| [206-vision-paddlegan-anime](notebooks/206-vision-paddlegan-anime/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/206-vision-paddlegan-anime/206-vision-paddlegan-anime.ipynb) | Turn an image into anime using a GAN | <img src="https://user-images.githubusercontent.com/15709723/127788059-1f069ae1-8705-4972-b50e-6314a6f36632.jpeg" width=100>→<img src="https://user-images.githubusercontent.com/15709723/125184441-b4584e80-e1d2-11eb-8964-d8131cd97409.png" width=100> |
| [207-vision-paddlegan-superresolution](notebooks/207-vision-paddlegan-superresolution/)<br> | Upscale small images with superresolution using a PaddleGAN model| |
| [208-optical-character-recognition](notebooks/208-optical-character-recognition/)<br> | Annotate text on images using text recognition resnet | <img src="https://user-images.githubusercontent.com/36741649/129315292-a37266dc-dfb2-4749-bca5-2ac9c1e93d64.jpg" width=225> |
| [209-handwritten-ocr](notebooks/209-handwritten-ocr/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F209-handwritten-ocr%2F209-handwritten-ocr.ipynb) | OCR for handwritten simplified Chinese and Japanese | <img width="425" alt="handwritten_simplified_chinese_test" src="https://user-images.githubusercontent.com/36741649/132660640-da2211ec-c389-450e-8980-32a75ed14abb.png"> <br> 的人不一了是他有为在责新中任自之我们 |
| [210-slowfast-video-recognition](notebooks/210-slowfast-video-recognition/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F210-slowfast-video-recognition%2F210-slowfast-video-recognition.ipynb) | Video Recognition using SlowFast and OpenVINO™ | <img src=https://github.com/facebookresearch/SlowFast/raw/main/demo/ava_demo.gif width=225> | 
| [211-speech-to-text](notebooks/211-speech-to-text/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F211-speech-to-text%2F211-speech-to-text.ipynb) | Run inference on speech-to-text recognition model | <img src="https://user-images.githubusercontent.com/36741649/140987347-279de058-55d7-4772-b013-0f2b12deaa61.png" width=225>|
| [212-pyannote-speaker-diarization](notebooks/212-pyannote-speaker-diarization/)<br> | Run inference on speaker diarization pipeline | <img src="https://user-images.githubusercontent.com/29454499/218432101-0bd0c424-e1d8-46af-ba1d-ee29ed6d1229.png" width=225>|
| [213-question-answering](notebooks/213-question-answering/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F213-question-answering%2F213-question-answering.ipynb) | Answer your questions basing on a context | <img src="https://user-images.githubusercontent.com/4547501/152571639-ace628b2-e3d2-433e-8c28-9a5546d76a86.gif" width=225> |
| [214-grammar-correction](notebooks/214-grammar-correction/) | Grammatical Error Correction with OpenVINO | **Input text**: `I'm working in campany for last 2 yeas.` <br> **Generated text**: `I'm working in a company for the last 2 years.` |
| [215-image-inpainting](notebooks/215-image-inpainting/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F215-image-inpainting%2F215-image-inpainting.ipynb)| Fill missing pixels with image in-painting | <img src="https://user-images.githubusercontent.com/4547501/167121084-ec58fbdb-b269-4de2-9d4c-253c5b95de1e.png" width=225> |
| [216-attention-center](notebooks/216-attention-center/)<br>| The attention center model with OpenVINO™ |  |
| [217-vision-deblur](notebooks/217-vision-deblur/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F217-vision-deblur%2F217-vision-deblur.ipynb) | Deblur Images with DeblurGAN-v2 | <img src="https://user-images.githubusercontent.com/41332813/158430181-05d07f42-cdb8-4b7a-b7dc-e7f7d9391877.png" width=225> |
| [218-vehicle-detection-and-recognition](notebooks/218-vehicle-detection-and-recognition/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F218-vehicle-detection-and-recognition%2F218-vehicle-detection-and-recognition.ipynb) | Use pre-trained models to detect and recognize vehicles and their attributes with OpenVINO | <img src = "https://user-images.githubusercontent.com/47499836/163544861-fa2ad64b-77df-4c16-b065-79183e8ed964.png" width=225> |
| [219-knowledge-graphs-conve](notebooks/219-knowledge-graphs-conve/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F219-knowledge-graphs-conve%2F219-knowledge-graphs-conve.ipynb) | Optimize the knowledge graph embeddings model (ConvE) with OpenVINO ||
| [220-books-alignment-labse](notebooks/220-cross-lingual-books-alignment/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F220-cross-lingual-books-alignment%2F220-cross-lingual-books-alignment.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/220-cross-lingual-books-alignment/220-cross-lingual-books-alignment.ipynb)| Cross-lingual Books Alignment With Transformers and OpenVINO™ |  |
| [221-machine-translation](notebooks/221-machine-translation)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F221-machine-translation%2F221-machine-translation.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/221-machine-translation/221-machine-translation.ipynb) | Real-time translation from English to German |  |
| [222-vision-image-colorization](notebooks/222-vision-image-colorization/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F222-vision-image-colorization%2F222-vision-image-colorization.ipynb) | Use pre-trained models to colorize black \& white images using OpenVINO | <img src = "https://user-images.githubusercontent.com/18904157/166343139-c6568e50-b856-4066-baef-5cdbd4e8bc18.png" width=225> |
| [223-text-prediction](notebooks/223-text-prediction/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/223-text-prediction/223-text-prediction.ipynb) | Use pretrained models to perform text prediction on an input sequence | <img src=https://user-images.githubusercontent.com/91228207/185105225-0f996b0b-0a3b-4486-872d-364ac6fab68b.png  width=225> |
| [224-3D-segmentation-point-clouds](notebooks/224-3D-segmentation-point-clouds/)<br> | Process point cloud data and run 3D Part Segmentation with OpenVINO | <img src = "https://user-images.githubusercontent.com/91237924/185752178-3882902c-907b-4614-b0e6-ea1de08bf3ef.png" width=225> |
| [225-stable-diffusion-text-to-image](notebooks/225-stable-diffusion-text-to-image)<br> | Text-to-image generation with Stable Diffusion method | <img src=https://user-images.githubusercontent.com/29454499/216524089-ed671fc7-a78b-42bf-aa96-9f7c791a9419.png width=225>|
| [226-yolov7-optimization](notebooks/226-yolov7-optimization/)<br> | Optimize YOLOv7 using NNCF PTQ API | <img src=https://raw.githubusercontent.com/WongKinYiu/yolov7/main/figure/horses_prediction.jpg  width=225> |
| [227-whisper-subtitles-generation](notebooks/227-whisper-subtitles-generation/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/227-whisper-subtitles-generation/227-whisper-subtitles-generation.ipynb) | Generate subtitles for video with OpenAI Whisper and OpenVINO | <img src=https://user-images.githubusercontent.com/29454499/204548693-1304ef33-c790-490d-8a8b-d5766acb6254.png  width=225> |
| [228-clip-zero-shot-image-classification](notebooks/228-clip-zero-shot-image-classification)<br> | Perform Zero-shot Image Classification with CLIP and OpenVINO | <img src=https://user-images.githubusercontent.com/29454499/207795060-437b42f9-e801-4332-a91f-cc26471e5ba2.png  width=500> |
| [229-distilbert-sequence-classification](notebooks/229-distilbert-sequence-classification/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F229-distilbert-sequence-classification%2F229-distilbert-sequence-classification.ipynb) | Sequence Classification with OpenVINO | <img src = "https://user-images.githubusercontent.com/95271966/206130638-d9847414-357a-4c79-9ca7-76f4ae5a6d7f.png" width=225> |
| [230-yolov8-optimization](notebooks/230-yolov8-optimization/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/230-yolov8-optimization/230-yolov8-optimization.ipynb) | Optimize YOLOv8 using NNCF PTQ API | <img src = "https://user-images.githubusercontent.com/29454499/212105105-f61c8aab-c1ff-40af-a33f-d0ed1fccc72e.png" width=225> |
| [231-instruct-pix2pix-image-editing](notebooks/231-instruct-pix2pix-image-editing/)<br>| Image editing with InstructPix2Pix | <img src=https://user-images.githubusercontent.com/29454499/219943222-d46a2e2d-d348-4259-8431-37cf14727eda.png width=225> |
| [232-clip-language-saliency-map](notebooks/232-clip-language-saliency-map/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/232-clip-language-saliency-map/232-clip-language-saliency-map.ipynb) | Language-Visual Saliency with CLIP and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/218967961-9858efd5-fff2-4eb0-bde9-60852f4b31cb.JPG width=225> | 
| [233-blip-visual-language-processing](notebooks/233-blip-visual-language-processing/)<br>| Visual Question Answering and Image Captioning using BLIP and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/221933762-4ff32ecb-5e5d-4484-80e1-e9396cb3c511.png width=225> |
| [234-encodec-audio-compression](notebooks/234-encodec-audio-compression/)<br>| Audio compression with EnCodec and OpenVINO™ | <img src=https://github.com/facebookresearch/encodec/raw/main/thumbnail.png width=225> |
| [235-controlnet-stable-diffusion](notebooks/235-controlnet-stable-diffusion/)<br>| A Text-to-Image Generation with ControlNet Conditioning and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/224541412-9d13443e-0e42-43f2-8210-aa31820c5b44.png width=225> |
| [236-stable-diffusion-v2](notebooks/236-stable-diffusion-v2/)<br>| Text-to-Image Generation and Infinite Zoom with Stable Diffusion v2 and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/228882108-25c1f65d-4c23-4e1d-8ba4-f6164280a3e3.gif width=225> |
| [237-segment-anything](notebooks/237-segment-anything/)<br>| Prompt based segmentation using Segment Anything and OpenVINO™.| <img src=https://user-images.githubusercontent.com/29454499/231468849-1cd11e68-21e2-44ed-8088-b792ef50c32d.png width=225> |
| [238-deep-floyd-if](notebooks/238-deeployd-if/)<br>| Text-to-Image Generation with DeepFloyd IF and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/241643886-dfcf3c48-8d50-4730-ae28-a21595d9504f.png width=225> |
| [239-image-bind](notebooks/239-image-bind/)<br>| Binding multimodal data using ImageBind and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/240364108-39868933-d221-41e6-9b2e-dac1b14ef32f.png width=225> |
| [240-dolly-2-instruction-following](notebooks/240-dolly-2-instruction-following/)<br>| Instruction following using Databricks Dolly 2.0 and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/237291423-022f07d2-966b-4be2-9a1c-98f1cf0691c2.png width=225> |
| [241-riffusion-text-to-music](notebooks/241-riffusion-text-to-music/)<br>| Text-to-Music generation using Riffusion and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/244291912-bbc6e08c-c0a9-41fe-bc2d-5f89a0d2463b.png width=225> | 
| [242-freevc-voice-conversion](notebooks/242-freevc-voice-conversion/)<br> | High-Quality Text-Free One-Shot Voice Conversion with FreeVC and OpenVINO™ ||
| [243-tflite-selfie-segmentation](notebooks/243-tflite-selfie-segmentation/) <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F243-tflite-selfie-segmentation%2F243-tflite-selfie-segmentation.ipynb)<br> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/243-tflite-selfie-segmentation/243-tflite-selfie-segmentation.ipynb)| Selfie Segmentation using TFLite and OpenVINO™ |  <img src="https://user-images.githubusercontent.com/29454499/251085926-14045ebc-273b-4ccb-b04f-82a3f7811b87.gif" width=400>|
| [244-named-entity-recognition](notebooks/244-named-entity-recognition/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/244-named-entity-recognition/244-named-entity-recognition.ipynb) | Named entity recognition with OpenVINO™ | |
| [245-typo-detector](notebooks/245-typo-detector/)<br>| English Typo Detection in sentences with OpenVINO™ | <img src=https://user-images.githubusercontent.com/80534358/224564463-ee686386-f846-4b2b-91af-7163586014b7.png   width=225> |
| [246-depth-estimation-videpth](notebooks/246-depth-estimation-videpth/)<br>| Monocular Visual-Inertial Depth Estimation with OpenVINO™ | <img src=https://raw.githubusercontent.com/alexklwong/void-dataset/master/figures/void_samples.png width=225> |
| [247-code-language-id](notebooks/247-code-language-id/247-code-language-id.ipynb)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F247-code-language-id%2F247-code-language-id.ipynb) | Identify the programming language used in an arbitrary code snippet | ||
| [248-stable-diffusion-xl](notebooks/248-stable-diffusion-xl/)<br> |  Image generation with Stable Diffusion XL and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/258651862-28b63016-c5ff-4263-9da8-73ca31100165.jpeg width=225>  |
| [249-oneformer-segmentation](notebooks/249-oneformer-segmentation/)<br> | Universal segmentation with OneFormer and OpenVINO™ | <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/76161256/258640713-f801bd09-e927-4abd-aa2f-9990de4caf8d.gif" width=225> |
| [250-music-generation](notebooks/250-music-generation/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F250-music-generation%2F250-music-generation.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/igor-davidyuk/openvino_notebooks/blob/musicgen/notebooks/250-music-generation/250-music-generation.ipynb) | Controllable Music Generation with MusicGen and OpenVINO™ | <img src="https://user-images.githubusercontent.com/76463150/260439306-81c81c8d-1f9c-41d0-b881-9491766def8e.png" width=225> |

<div id='-model-training'></div>

### 🏃 Model Training

Tutorials that include code to train neural networks.

| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [301-tensorflow-training-openvino](notebooks/301-tensorflow-training-openvino/) | Train a flower classification model from TensorFlow, then convert to OpenVINO IR | <img src="https://user-images.githubusercontent.com/15709723/127779607-8fa34947-1c35-4260-8d04-981c41a2a2cc.png" width=390> |
| [301-tensorflow-training-openvino-nncf](notebooks/301-tensorflow-training-openvino/) | Use Neural Network Compression Framework (NNCF) to quantize model from TensorFlow | |
| [302-pytorch-quantization-aware-training](notebooks/302-pytorch-quantization-aware-training/) | Use Neural Network Compression Framework (NNCF) to quantize PyTorch model | |
| [305-tensorflow-quantization-aware-training](notebooks/305-tensorflow-quantization-aware-training/)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/305-tensorflow-quantization-aware-training/305-tensorflow-quantization-aware-training.ipynb) | Use Neural Network Compression Framework (NNCF) to quantize TensorFlow model | |

<div id='-live-demos'></div>

### 📺 Live Demos

Live inference demos that run on a webcam or video files.

| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [401-object-detection-webcam](notebooks/401-object-detection-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F401-object-detection-webcam%2F401-object-detection.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/401-object-detection-webcam/401-object-detection.ipynb) | Object detection with a webcam or video file  | <img src="https://user-images.githubusercontent.com/4547501/141471665-82b28c86-cf64-4bfe-98b3-c314658f2d96.gif" width=225> |
| [402-pose-estimation-webcam](notebooks/402-pose-estimation-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F402-pose-estimation-webcam%2F402-pose-estimation.ipynb) | Human pose estimation with a webcam or video file | <img src="https://user-images.githubusercontent.com/4547501/138267961-41d754e7-59db-49f6-b700-63c3a636fad7.gif" width=225> |
| [403-action-recognition-webcam](notebooks/403-action-recognition-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F403-action-recognition-webcam%2F403-action-recognition-webcam.ipynb) | Human action recognition with a webcam or video file | <img src="https://user-images.githubusercontent.com/10940214/151552326-642d6e49-f5a0-4fc1-bf14-ae3f457e1fec.gif" width=225> |
| [404-style-transfer-webcam](notebooks/404-style-transfer-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F404-style-transfer-webcam%2F404-style-transfer.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/404-style-transfer-webcam/404-style-transfer.ipynb) | Style Transfer with a webcam or video file | <img src="https://user-images.githubusercontent.com/109281183/203772234-f17a0875-b068-43ef-9e77-403462fde1f5.gif" width=250> |
| [405-paddle-ocr-webcam](notebooks/405-paddle-ocr-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F405-paddle-ocr-webcam%2F405-paddle-ocr-webcam.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/405-paddle-ocr-webcam/405-paddle-ocr-webcam.ipynb) | OCR with a webcam or video file | <img src="https://raw.githubusercontent.com/yoyowz/classification/master/images/paddleocr.gif" width=225> |
| [406-3D-pose-estimation-webcam](notebooks/406-3D-pose-estimation-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F406-3D-pose-estimation-webcam%2F406-3D-pose-estimation.ipynb) | 3D display of human pose estimation with a webcam or video file | <img src = "https://user-images.githubusercontent.com/42672437/183292131-576cc05a-a724-472c-8dc9-f6bc092190bf.gif" width=225> |
| [407-person-tracking-webcam](notebooks/407-person-tracking-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F407-person-tracking-webcam%2F407-person-tracking.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/407-person-tracking-webcam/407-person-tracking.ipynb) | Person tracking with a webcam or video file | <img src = "https://user-images.githubusercontent.com/91237924/210479548-b70dbbaa-5948-4e49-b48e-6cb6613226da.gif" width=225> |

If you run into issues, please check the [troubleshooting section](#-troubleshooting), [FAQs](#-faq) or start a GitHub [discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions).

Notebooks with ![binder logo](https://mybinder.org/badge_logo.svg) and ![colab logo](https://colab.research.google.com/assets/colab-badge.svg) buttons can be run without installing anything. [Binder](https://mybinder.org/) and [Google Colab](https://colab.research.google.com/) are free online services with limited resources. For the best performance, please follow the [Installation Guide](#-installation-guide) and run the notebooks locally.

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-system-requirements'></div>

## ⚙️ System Requirements

The notebooks run almost anywhere &mdash; your laptop, a cloud VM, or even a Docker container. The table below lists the supported operating systems and Python versions.

| Supported Operating System                                 | [Python Version (64-bit)](https://www.python.org/) |
| :--------------------------------------------------------- |:---------------------------------------------------|
| Ubuntu 20.04 LTS, 64-bit                                   | 3.8 - 3.10                                         |
| Ubuntu 22.04 LTS, 64-bit                                   | 3.8 - 3.10                                         |
| Red Hat Enterprise Linux 8, 64-bit                         | 3.8 - 3.10                                         |
| CentOS 7, 64-bit                                           | 3.8 - 3.10                                         |
| macOS 10.15.x versions or higher                           | 3.8 - 3.10                                         |
| Windows 10, 64-bit Pro, Enterprise or Education editions   | 3.8 - 3.10                                         |
| Windows Server 2016 or higher                              | 3.8 - 3.10                                         |

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)](#)
<div id='-run-the-notebooks'></div>

## 💻 Run the Notebooks

### To Launch a Single Notebook

If you wish to launch only one notebook, like the Monodepth notebook, run the command below.

```bash
jupyter 201-vision-monodepth.ipynb
```

### To Launch all Notebooks

```bash
jupyter lab notebooks
```

In your browser, select a notebook from the file browser in Jupyter Lab using the left sidebar. Each tutorial is located in a subdirectory within the `notebooks` directory.

<img src="https://user-images.githubusercontent.com/15709723/120527271-006fd200-c38f-11eb-9935-2d36d50bab9f.gif">

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-cleaning-up'></div>

## 🧹 Cleaning Up

<div id='-shut-down-jupyter-kernel' markdown="1">

1. Shut Down Jupyter Kernel

	To end your Jupyter session, press `Ctrl-c`. This will prompt you to `Shutdown this Jupyter server (y/[n])?` enter `y` and hit `Enter`.
</div>	
	
<div id='-deactivate-virtual-environment' markdown="1">

2. Deactivate Virtual Environment

	To deactivate your virtualenv, simply run `deactivate` from the terminal window where you activated `openvino_env`. This will deactivate your environment.

	To reactivate your environment, run `source openvino_env/bin/activate` on Linux or `openvino_env\Scripts\activate` on Windows, then type `jupyter lab` or `jupyter notebook` to launch the notebooks again.
</div>	

<div id='-delete-virtual-environment' markdown="1">

3. Delete Virtual Environment _(Optional)_

	To remove your virtual environment, simply delete the `openvino_env` directory:
</div>

<div id='-on-linux-and-macos' markdown="1">

  - On Linux and macOS:

	```bash
	rm -rf openvino_env
	```
</div>

<div id='-on-windows' markdown="1">

  - On Windows:

	```bash
	rmdir /s openvino_env
	```
</div>

<div id='-remove-openvino-env-kernel' markdown="1">

  - Remove `openvino_env` Kernel from Jupyter

	```bash
	jupyter kernelspec remove openvino_env
	```
</div>

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-troubleshooting'></div>

## ⚠️ Troubleshooting

If these tips do not solve your problem, please open a [discussion topic](https://github.com/openvinotoolkit/openvino_notebooks/discussions)
or create an [issue](https://github.com/openvinotoolkit/openvino_notebooks/issues)!

- To check some common installation problems, run `python check_install.py`. This script is located in the openvino_notebooks directory.
  Please run it after activating the `openvino_env` virtual environment.
- If you get an `ImportError`, double-check that you installed the Jupyter kernel. If necessary, choose the `openvino_env` kernel from the _Kernel->Change Kernel_ menu in Jupyter Lab or Jupyter Notebook.
- If OpenVINO is installed globally, do not run installation commands in a terminal where `setupvars.bat` or `setupvars.sh` are sourced.
- For Windows installation, it is recommended to use _Command Prompt (`cmd.exe`)_, not _PowerShell_.

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)](#-contributors)
<div id='-contributors'></div>

## 🧑‍💻 Contributors

<a href="https://github.com/openvinotoolkit/openvino_notebooks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openvinotoolkit/openvino_notebooks" />
</a>

Made with [`contrib.rocks`](https://contrib.rocks).

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-faq'></div>

## ❓ FAQ

* [Which devices does OpenVINO support?](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html#doxid-openvino-docs-o-v-u-g-supported-plugins-supported-devices)
* [What is the first CPU generation you support with OpenVINO?](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html)
* [Are there any success stories about deploying real-world solutions with OpenVINO?](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html)

---

\* Other names and brands may be claimed as the property of others.
