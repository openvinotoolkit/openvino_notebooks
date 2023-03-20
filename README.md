English | [简体中文](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README_cn.md)

<h1 align="center">📚 OpenVINO™ Notebooks</h1>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/LICENSE)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/nbval_precommit.yml/badge.svg?event=push)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/nbval_precommit.yml?query=event%3Apush)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml/badge.svg?event=push)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml?query=event%3Apush)

A collection of ready-to-run Jupyter notebooks for learning and experimenting with the OpenVINO™ Toolkit. The notebooks provide an introduction to OpenVINO basics and teach developers how to leverage our API for optimized deep learning inference.

**NOTE: The main branch of this repository was updated to support the new OpenVINO 2022.3 release.** To upgrade to the new release version, please run `pip install --upgrade -r requirements.txt` in your `openvino_env` virtual environment. If you need to install for the first time, see the [Installation Guide](#-installation-guide) section below. If you wish to use the previous Long Term Support (LTS) version of OpenVINO check out the [2021.4 branch](https://github.com/openvinotoolkit/openvino_notebooks/tree/2021.4). 

If you need help, please start a GitHub [Discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions).  

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

## Table of Contents

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

| [Windows](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) | [Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu) | [macOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS) | [Red Hat](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [CentOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [Azure ML](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML) | [Docker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Docker) | [Amazon SageMaker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/SageMaker)|
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |--------------------------------------------------------------------------- |
	
[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-getting-started'/>

## 🚀 Getting Started

The Jupyter notebooks are categorized into four classes, select one related to your needs or give them all a try. Good Luck! 

<div id='-first-steps'/>

### 💻 First steps

Brief tutorials that demonstrate how to use OpenVINO's Python API for inference.

| [001-hello-world](notebooks/001-hello-world/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F001-hello-world%2F001-hello-world.ipynb) | [002-openvino-api](notebooks/002-openvino-api/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F002-openvino-api%2F002-openvino-api.ipynb) | [003-hello-segmentation](notebooks/003-hello-segmentation/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F003-hello-segmentation%2F003-hello-segmentation.ipynb) | [004-hello-detection](notebooks/004-hello-detection/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F004-hello-detection%2F004-hello-detection.ipynb) | 
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |  
| Classify an image with OpenVINO | Learn the OpenVINO Python API | Semantic segmentation with OpenVINO | Text detection with OpenVINO  | 
| <img src="https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg" width=140> | <img src="https://user-images.githubusercontent.com/15709723/127787560-d8ec4d92-b4a0-411f-84aa-007e90faba98.png" width=250> | <img src="https://user-images.githubusercontent.com/15709723/128290691-e2eb875c-775e-4f4d-a2f4-15134044b4bb.png" width=150> | <img src="https://user-images.githubusercontent.com/36741649/128489933-bf215a3f-06fa-4918-8833-cb0bf9fb1cc7.jpg" width=150>  | 

<div id='-convert--optimize'/>

### ⌚ Convert & Optimize 

Tutorials that explain how to optimize and quantize models with OpenVINO tools.
	
| [101-tensorflow-to-openvino](notebooks/101-tensorflow-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F101-tensorflow-to-openvino%2F101-tensorflow-to-openvino.ipynb) |  [102-pytorch-onnx-to-openvino](notebooks/102-pytorch-onnx-to-openvino/) | [103-paddle-to-openvino](notebooks/103-paddle-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F103-paddle-to-openvino%2F103-paddle-to-openvino-classification.ipynb) | [104-model-tools](notebooks/104-model-tools/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F104-model-tools%2F104-model-tools.ipynb) | 
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |  
| Convert TensorFlow models to OpenVINO IR | Convert PyTorch models to OpenVINO IR | Convert PaddlePaddle models to OpenVINO IR | Download, convert and benchmark models from Open Model Zoo  | 
| <img src="https://user-images.githubusercontent.com/15709723/127779167-9d33dcc6-9001-4d74-a089-8248310092fe.png" width=250> | <img src="https://user-images.githubusercontent.com/15709723/127779246-32e7392b-2d72-4a7d-b871-e79e7bfdd2e9.png" width=300 > | <img src="https://user-images.githubusercontent.com/15709723/127779326-dc14653f-a960-4877-b529-86908a6f2a61.png" width=300>  | <img src="https://user-images.githubusercontent.com/10940214/157541917-c5455105-b0d9-4adf-91a7-fbc142918015.png" width=150>  |
	
More amazing notebooks here! 

<p>
<details>
<summary> Click here to show complete list!  </summary> 

| Notebook | Description | 
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | 
| [101-tensorflow-to-openvino](notebooks/101-tensorflow-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F101-tensorflow-to-openvino%2F101-tensorflow-to-openvino.ipynb) | Convert TensorFlow models to OpenVINO IR | 
| [102-pytorch-onnx-to-openvino](notebooks/102-pytorch-onnx-to-openvino/) | Convert PyTorch models to OpenVINO IR | 
| [103-paddle-to-openvino](notebooks/103-paddle-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F103-paddle-to-openvino%2F103-paddle-to-openvino-classification.ipynb) | Convert PaddlePaddle models to OpenVINO IR | 
| [104-model-tools](notebooks/104-model-tools/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F104-model-tools%2F104-model-tools.ipynb) | Download, convert and benchmark models from Open Model Zoo | 
| [105-language-quantize-bert](notebooks/105-language-quantize-bert/) | Optimize and quantize a pre-trained BERT model |
| [106-auto-device](notebooks/106-auto-device/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F106-auto-device%2F106-auto-device.ipynb) | Demonstrate how to use AUTO Device |
| [107-speech-recognition-quantization](notebooks/107-speech-recognition-quantization/) | Optimize and quantize a pre-trained speech recognition models |
| [110-ct-segmentation-quantize](notebooks/110-ct-segmentation-quantize/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F110-ct-segmentation-quantize%2F110-ct-scan-live-inference.ipynb) | Quantize a kidney segmentation model and show live inference | 
| [111-detection-quantization](notebooks/111-detection-quantization/)<br> | Quantize an object detection model | 
| [112-pytorch-post-training-quantization-nncf](notebooks/112-pytorch-post-training-quantization-nncf/) | Use Neural Network Compression Framework (NNCF) to quantize PyTorch model in post-training mode (without model fine-tuning)| 
| [113-image-classification-quantization](notebooks/113-image-classification-quantization/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F113-image-classification-quantization%2F113-image-classification-quantization.ipynb) | Quantize mobilenet image classification |
| [114-quantization-simplified-mode](notebooks/114-quantization-simplified-mode/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F114-quantization-simplified-mode%2F114-quantization-simplified-mode.ipynb) | Quantize Image Classification Models with POT in Simplified Mode|
| [115-async-api](notebooks/115-async-api/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F115-async-api%2F115-async-api.ipynb) | Use Asynchronous Execution to Improve Data Pipelining| |
| [116-sparsity-optimization](notebooks/116-sparsity-optimization/)<br> | Improve performance of sparse Transformer models |
| [117-model-server](notebooks/117-model-server/)| Introduction to model serving with OpenVINO™ Model Server (OVMS) |
| [118-optimize-preprocessing](notebooks/118-optimize-preprocessing/)| Improve performance of image preprocessing step |

</details>
</p>

<div id='-model-demos'/>

### 🎯 Model Demos

Demos that demonstrate inference on a particular model.
	
| [225-stable-diffusion-text-to-image](notebooks/225-stable-diffusion-text-to-image/)                                          | [227-whisper-subtitles-generation](notebooks/227-whisper-subtitles-generation/)| [230-yolov8-optimization](notebooks/230-yolov8-optimization/) |  [231-instruct-pix2pix-image-editing](notebooks/231-instruct-pix2pix-image-editing/)|
|------------------------------------------------------------------------------------------------------------------------------| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | 
| Text-to-image generation with Stable Diffusion method | Generate subtitles for video with OpenAI Whisper and OpenVINO | Optimize YOLOv8 using NNCF PTQ API | Image editing with InstructPix2Pix |
| <img src=https://user-images.githubusercontent.com/29454499/216524089-ed671fc7-a78b-42bf-aa96-9f7c791a9419.png width=225>| <img src=https://user-images.githubusercontent.com/29454499/204548693-1304ef33-c790-490d-8a8b-d5766acb6254.png  width=225> | <img src = "https://user-images.githubusercontent.com/29454499/212105105-f61c8aab-c1ff-40af-a33f-d0ed1fccc72e.png" width=225> | <img src=https://user-images.githubusercontent.com/29454499/219943222-d46a2e2d-d348-4259-8431-37cf14727eda.png width=750> |
	
More amazing notebooks here! 

<p>
<details>
<summary> Click here to show complete list! </summary>
	
	
| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [201-vision-monodepth](notebooks/201-vision-monodepth/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F201-vision-monodepth%2F201-vision-monodepth.ipynb) | Monocular depth estimation with images and video | <img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif" width=250> |
| [202-vision-superresolution-image](notebooks/202-vision-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-image.ipynb) | Upscale raw images with a super resolution model | <img src="https://user-images.githubusercontent.com/36741649/170005347-e4409f9e-ec34-416b-afdf-a9d8185929ca.jpg" width="70">→<img src="https://user-images.githubusercontent.com/36741649/170005347-e4409f9e-ec34-416b-afdf-a9d8185929ca.jpg" width="130"> |
| [202-vision-superresolution-video](notebooks/202-vision-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-video.ipynb) | Turn 360p into 1080p video using a super resolution model | <img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width=80>→<img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width="125"> |
| [203-meter-reader](notebooks/203-meter-reader/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F203-meter-reader%2F203-meter-reader.ipynb) | PaddlePaddle pre-trained models to read industrial meter's value | <img src="https://user-images.githubusercontent.com/91237924/166135627-194405b0-6c25-4fd8-9ad1-83fb3a00a081.jpg" width=225> |
| [205-vision-background-removal](notebooks/205-vision-background-removal/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F205-vision-background-removal%2F205-vision-background-removal.ipynb) | Remove and replace the background in an image using salient object detection | <img src="https://user-images.githubusercontent.com/15709723/125184237-f4b6cd00-e1d0-11eb-8e3b-d92c9a728372.png" width=455> |
| [206-vision-paddlegan-anime](notebooks/206-vision-paddlegan-anime/)<br> | Turn an image into anime using a GAN | <img src="https://user-images.githubusercontent.com/15709723/127788059-1f069ae1-8705-4972-b50e-6314a6f36632.jpeg" width=100>→<img src="https://user-images.githubusercontent.com/15709723/125184441-b4584e80-e1d2-11eb-8964-d8131cd97409.png" width=100> |
| [207-vision-paddlegan-superresolution](notebooks/207-vision-paddlegan-superresolution/)<br> | Upscale small images with superresolution using a PaddleGAN model| |
| [208-optical-character-recognition](notebooks/208-optical-character-recognition/)<br> | Annotate text on images using text recognition resnet | <img src="https://user-images.githubusercontent.com/36741649/129315292-a37266dc-dfb2-4749-bca5-2ac9c1e93d64.jpg" width=225> |
| [209-handwritten-ocr](notebooks/209-handwritten-ocr/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F209-handwritten-ocr%2F209-handwritten-ocr.ipynb) | OCR for handwritten simplified Chinese and Japanese | <img width="425" alt="handwritten_simplified_chinese_test" src="https://user-images.githubusercontent.com/36741649/132660640-da2211ec-c389-450e-8980-32a75ed14abb.png"> <br> 的人不一了是他有为在责新中任自之我们 |
| [211-speech-to-text](notebooks/211-speech-to-text/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F211-speech-to-text%2F211-speech-to-text.ipynb) | Run inference on speech-to-text recognition model | <img src="https://user-images.githubusercontent.com/36741649/140987347-279de058-55d7-4772-b013-0f2b12deaa61.png" width=225>|
| [212-pyannote-speaker-diarization](notebooks/212-pyannote-speaker-diarization/)<br> | Run inference on speaker diarization pipeline | <img src="https://user-images.githubusercontent.com/29454499/218432101-0bd0c424-e1d8-46af-ba1d-ee29ed6d1229.png" width=225>|
| [213-question-answering](notebooks/213-question-answering/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F213-question-answering%2F213-question-answering.ipynb) | Answer your questions basing on a context | <img src="https://user-images.githubusercontent.com/4547501/152571639-ace628b2-e3d2-433e-8c28-9a5546d76a86.gif" width=225> |
| [214-grammar-correction](notebooks/214-grammar-correction/) | Grammatical Error Correction with OpenVINO | **input text**: I'm working in campany for last 2 yeas <br> **Generated text**: I'm working in a company for the last 2 years. |
| [215-image-inpainting](notebooks/215-image-inpainting/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F215-image-inpainting%2F215-image-inpainting.ipynb)| Fill missing pixels with image in-painting | <img src="https://user-images.githubusercontent.com/4547501/167121084-ec58fbdb-b269-4de2-9d4c-253c5b95de1e.png" width=225> |
| [216-license-plate-recognition](notebooks/216-license-plate-recognition/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F216-license-plate-recognition%2F216-license-plate-recognition.ipynb) | Recognize Chinese license plates in traffic | <img src="https://user-images.githubusercontent.com/70456146/162759539-4a0a996f-dabe-40ea-98d6-85b4dce8511d.png" width=225> |
| [217-vision-deblur](notebooks/217-vision-deblur/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F217-vision-deblur%2F217-vision-deblur.ipynb) | Deblur Images with DeblurGAN-v2 | <img src="https://user-images.githubusercontent.com/41332813/158430181-05d07f42-cdb8-4b7a-b7dc-e7f7d9391877.png" width=225> |
| [218-vehicle-detection-and-recognition](notebooks/218-vehicle-detection-and-recognition/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F218-vehicle-detection-and-recognition%2F218-vehicle-detection-and-recognition.ipynb) | Use pre-trained models to detect and recognize vehicles and their attributes with OpenVINO | <img src = "https://user-images.githubusercontent.com/47499836/163544861-fa2ad64b-77df-4c16-b065-79183e8ed964.png" width=225> |
| [219-knowledge-graphs-conve](notebooks/219-knowledge-graphs-conve/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F219-knowledge-graphs-conve%2F219-knowledge-graphs-conve.ipynb) | Optimize the knowledge graph embeddings model (ConvE) with OpenVINO ||
| [220-yolov5-accuracy-check-and-quantization](notebooks/220-yolov5-accuracy-check-and-quantization)<br> | Quantize the Ultralytics YOLOv5 model and check accuracy using the OpenVINO POT API | <img src = "https://user-images.githubusercontent.com/44352144/177097174-cfe78939-e946-445e-9fce-d8897417ef8e.png"  width=225> |
| [221-machine-translation](notebooks/221-machine-translation)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F221-machine-translation%2F221-machine-translation.ipynb) | Real-time translation from English to German |  |
| [222-vision-image-colorization](notebooks/222-vision-image-colorization/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F222-vision-image-colorization%2F222-vision-image-colorization.ipynb) | Use pre-trained models to colorize black \& white images using OpenVINO | <img src = "https://user-images.githubusercontent.com/18904157/166343139-c6568e50-b856-4066-baef-5cdbd4e8bc18.png" width=225> |
| [223-gpt2-text-prediction](notebooks/223-gpt2-text-prediction/)<br> | Use GPT-2 to perform text prediction on an input sequence | <img src=https://user-images.githubusercontent.com/91228207/185105225-0f996b0b-0a3b-4486-872d-364ac6fab68b.png  width=225> |
| [224-3D-segmentation-point-clouds](notebooks/224-3D-segmentation-point-clouds/)<br> | Process point cloud data and run 3D Part Segmentation with OpenVINO | <img src = "https://user-images.githubusercontent.com/91237924/185752178-3882902c-907b-4614-b0e6-ea1de08bf3ef.png" width=225> |
| [225-stable-diffusion-text-to-image](notebooks/225-stable-diffusion-text-to-image)<br> | Text-to-image generation with Stable Diffusion method | <img src=https://user-images.githubusercontent.com/29454499/216524089-ed671fc7-a78b-42bf-aa96-9f7c791a9419.png width=225>|
| [226-yolov7-optimization](notebooks/226-yolov7-optimization/)<br> | Optimize YOLOv7 using NNCF PTQ API | <img src=https://raw.githubusercontent.com/WongKinYiu/yolov7/main/figure/horses_prediction.jpg  width=225> |
| [227-whisper-subtitles-generation](notebooks/227-whisper-subtitles-generation/)<br> | Generate subtitles for video with OpenAI Whisper and OpenVINO | <img src=https://user-images.githubusercontent.com/29454499/204548693-1304ef33-c790-490d-8a8b-d5766acb6254.png  width=225> |
| [228-clip-zero-shot-image-classification](notebooks/228-clip-zero-shot-image-classification)<br> | Perform Zero-shot Image Classification with CLIP and OpenVINO | <img src=https://user-images.githubusercontent.com/29454499/207795060-437b42f9-e801-4332-a91f-cc26471e5ba2.png  width=500> |
| [229-distilbert-sequence-classification](notebooks/229-distilbert-sequence-classification/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F229-distilbert-sequence-classification%2F229-distilbert-sequence-classification.ipynb) | Sequence Classification with OpenVINO | <img src = "https://user-images.githubusercontent.com/95271966/206130638-d9847414-357a-4c79-9ca7-76f4ae5a6d7f.png" width=225> |
| [230-yolov8-optimization](notebooks/230-yolov8-optimization/)<br> | Optimize YOLOv8 using NNCF PTQ API | <img src = "https://user-images.githubusercontent.com/29454499/212105105-f61c8aab-c1ff-40af-a33f-d0ed1fccc72e.png" width=225> |
|[231-instruct-pix2pix-image-editing](notebooks/231-instruct-pix2pix-image-editing/)<br>| Image editing with InstructPix2Pix | <img src=https://user-images.githubusercontent.com/29454499/219943222-d46a2e2d-d348-4259-8431-37cf14727eda.png width=225> |
|[232-clip-language-saliency-map](notebooks/232-clip-language-saliency-map/)<br>| Language-Visual Saliency with CLIP and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/218967961-9858efd5-fff2-4eb0-bde9-60852f4b31cb.JPG width=225> | 
|[233-blip-visual-language-processing](notebook/233-blip-visual-language-processingp/)<br>| Visual Question Answering and Image Captioning using BLIP and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/221933762-4ff32ecb-5e5d-4484-80e1-e9396cb3c511.png width=225> |
|[234-encodec-audio-compression](notebooks/234-encodec-audio-compression/)<br>| # Audio compression with EnCodec and OpenVINO™ | <img src=https://github.com/facebookresearch/encodec/raw/main/thumbnail.png width=225> |
|[235-controlnet-stable-diffusion](notebooks/235-controlnet-stable-diffusion/)<br>| # A Text-to-Image Generation with ControlNet Conditioning and OpenVINO™ | <img src=https://user-images.githubusercontent.com/29454499/224541412-9d13443e-0e42-43f2-8210-aa31820c5b44.png width=225> |
|[236-distillbert-question-answering](236-distillbert-question-answering/)<br>| Question Answering with DistillBERT and OpenVINO™ | <img src=https://raw.githubusercontent.com/Karthik-Bhaskar/Context-Based-Question-Answering/f2e0bbc03003aae65f4cabddecd5cd9fcdbfb333/docs_resources/CBQA_social_preview.png width=225> |
</details>
</p>

<div id='-model-training'/>

### 🏃 Model Training

Tutorials that include code to train neural networks.
| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [301-tensorflow-training-openvino](notebooks/301-tensorflow-training-openvino/) | Train a flower classification model from TensorFlow, then convert to OpenVINO IR | <img src="https://user-images.githubusercontent.com/15709723/127779607-8fa34947-1c35-4260-8d04-981c41a2a2cc.png" width=390> |
| [301-tensorflow-training-openvino-pot](notebooks/301-tensorflow-training-openvino/) | Use Post-training Optimization Tool (POT) to quantize the flowers model | |
| [302-pytorch-quantization-aware-training](notebooks/302-pytorch-quantization-aware-training/) | Use Neural Network Compression Framework (NNCF) to quantize PyTorch model | |
| [305-tensorflow-quantization-aware-training](notebooks/305-tensorflow-quantization-aware-training/) | Use Neural Network Compression Framework (NNCF) to quantize TensorFlow model | |

<div id='-live-demos'/>

### 📺 Live Demos
Live inference demos that run on a webcam or video files.
	
| [401-object-detection-webcam](notebooks/401-object-detection-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F401-object-detection-webcam%2F401-object-detection.ipynb) | [403-action-recognition-webcam](notebooks/403-action-recognition-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F403-action-recognition-webcam%2F403-action-recognition-webcam.ipynb) | [404-style-transfer-webcam](notebooks/404-style-transfer-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F404-style-transfer-webcam%2F404-style-transfer.ipynb) | [405-paddle-ocr-webcam](notebooks/405-paddle-ocr-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F405-paddle-ocr-webcam%2F405-paddle-ocr-webcam.ipynb) |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
|Object detection with a webcam or video file |  Human action recognition with a webcam or video file | Style transfer with a webcam or video file| OCR with a webcam or video file |
| <img src="https://user-images.githubusercontent.com/4547501/141471665-82b28c86-cf64-4bfe-98b3-c314658f2d96.gif" width=150> | <img src="https://user-images.githubusercontent.com/10940214/151552326-642d6e49-f5a0-4fc1-bf14-ae3f457e1fec.gif" width=200> | <img src="https://user-images.githubusercontent.com/109281183/203772234-f17a0875-b068-43ef-9e77-403462fde1f5.gif" width=300> | <img src="https://raw.githubusercontent.com/yoyowz/classification/master/images/paddleocr.gif" width=230> |
	
More amazing notebooks here! 

<p>
<details>
<summary> Click here to show complete list! </summary>
	
	
| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [401-object-detection-webcam](notebooks/401-object-detection-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F401-object-detection-webcam%2F401-object-detection.ipynb) | Object detection with a webcam or video file  | <img src="https://user-images.githubusercontent.com/4547501/141471665-82b28c86-cf64-4bfe-98b3-c314658f2d96.gif" width=225> |
| [402-pose-estimation-webcam](notebooks/402-pose-estimation-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F402-pose-estimation-webcam%2F402-pose-estimation.ipynb) | Human pose estimation with a webcam or video file | <img src="https://user-images.githubusercontent.com/4547501/138267961-41d754e7-59db-49f6-b700-63c3a636fad7.gif" width=225> |
| [403-action-recognition-webcam](notebooks/403-action-recognition-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F403-action-recognition-webcam%2F403-action-recognition-webcam.ipynb) | Human action recognition with a webcam or video file | <img src="https://user-images.githubusercontent.com/10940214/151552326-642d6e49-f5a0-4fc1-bf14-ae3f457e1fec.gif" width=225> |
| [404-style-transfer-webcam](notebooks/404-style-transfer-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F404-style-transfer-webcam%2F404-style-transfer.ipynb) | Style Transfer with a webcam or video file | <img src="https://user-images.githubusercontent.com/109281183/203772234-f17a0875-b068-43ef-9e77-403462fde1f5.gif" width=250> |
| [405-paddle-ocr-webcam](notebooks/405-paddle-ocr-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F405-paddle-ocr-webcam%2F405-paddle-ocr-webcam.ipynb) | OCR with a webcam or video file | <img src="https://raw.githubusercontent.com/yoyowz/classification/master/images/paddleocr.gif" width=225> |
| [406-3D-pose-estimation-webcam](notebooks/406-3D-pose-estimation-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks.git/main?labpath=notebooks%2F406-3D-pose-estimation-webcam%2F406-3D-pose-estimation.ipynb) | 3D display of human pose estimation with a webcam or video file | <img src = "https://user-images.githubusercontent.com/42672437/183292131-576cc05a-a724-472c-8dc9-f6bc092190bf.gif" width=225> |
| [407-person-tracking-webcam](notebooks/407-person-tracking-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F407-person-tracking-webcam%2F407-person-tracking.ipynb) | Person tracking with a webcam or video file | <img src = "https://user-images.githubusercontent.com/91237924/210479548-b70dbbaa-5948-4e49-b48e-6cb6613226da.gif" width=225> |


</details>
</p>

If you run into issues, please check the [troubleshooting section](#-troubleshooting), [FAQs](#-faq) or start a GitHub [discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions). 

Notebooks with a ![binder logo](https://mybinder.org/badge_logo.svg) button can be run without installing anything. [Binder](https://mybinder.org/) is a free online service with limited resources. For the best performance, please follow the [Installation Guide](#-installation-guide) and run the notebooks locally.

You will have a lot of fun with this section:

| [Vision-monodepth](notebooks/201-vision-monodepth/)  | [Object-detection-webcam](notebooks/401-object-detection-webcam/) | [Pose-estimation-webcam](notebooks/402-pose-estimation-webcam/) | [Action-recognition-webcam](notebooks/403-action-recognition-webcam/) | 
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | 
| <img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif" width=225> | <img src="https://user-images.githubusercontent.com/4547501/141471665-82b28c86-cf64-4bfe-98b3-c314658f2d96.gif" width=140> | <img src="https://user-images.githubusercontent.com/4547501/138267961-41d754e7-59db-49f6-b700-63c3a636fad7.gif" width=185>  | <img src="https://user-images.githubusercontent.com/10940214/151552326-642d6e49-f5a0-4fc1-bf14-ae3f457e1fec.gif" width=185> | 


[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-system-requirements'/>

## ⚙️ System Requirements

The notebooks run almost anywhere &mdash; your laptop, a cloud VM, or even a Docker container. The table below lists the supported operating systems and Python versions.

| Supported Operating System                                 | [Python Version (64-bit)](https://www.python.org/) |
| :--------------------------------------------------------- | :------------------------------------------------- |
| Ubuntu 20.04 LTS, 64-bit                                   | 3.7, 3.8, 3.9, 3.10                                |
| Ubuntu 22.04 LTS, 64-bit                                   | 3.7, 3.8, 3.9, 3.10                                |
| Red Hat Enterprise Linux 8, 64-bit                         | 3.8, 3.9, 3.10                                     |
| CentOS 7, 64-bit                                           | 3.7, 3.8, 3.9, 3.10                                |
| macOS 10.15.x versions or higher                           | 3.7, 3.8, 3.9, 3.10                                |
| Windows 10, 64-bit Pro, Enterprise or Education editions   | 3.7, 3.8, 3.9, 3.10                                |
| Windows Server 2016 or higher                              | 3.7, 3.8, 3.9, 3.10                                |

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)](#)
<div id='-run-the-notebooks'/>

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
<div id='-cleaning-up'/>

## 🧹 Cleaning Up

<p>
<details>
<summary>Shut Down Jupyter Kernel</summary>

To end your Jupyter session, press `Ctrl-c`. This will prompt you to `Shutdown this Jupyter server (y/[n])?` enter `y` and hit `Enter`.
</details>
</p>	
	
<p>
<details>
<summary>Deactivate Virtual Environment</summary>

To deactivate your virtualenv, simply run `deactivate` from the terminal window where you activated `openvino_env`. This will deactivate your environment.

To reactivate your environment, run `source openvino_env/bin/activate` on Linux or `openvino_env\Scripts\activate` on Windows, then type `jupyter lab` or `jupyter notebook` to launch the notebooks again.
</details>
</p>	
	
<p>
<details>
<summary>Delete Virtual Environment _(Optional)_</summary>

To remove your virtual environment, simply delete the `openvino_env` directory:
</details>
</p>	
	
<p>
<details>
<summary>On Linux and macOS:</summary>

```bash
rm -rf openvino_env
```
</details>
</p>

<p>
<details>
<summary>On Windows:</summary>

```bash
rmdir /s openvino_env
```
</details>
</p>

<p>
<details>
<summary>Remove openvino_env Kernel from Jupyter</summary>

```bash
jupyter kernelspec remove openvino_env
```
</details>
</p>

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-troubleshooting'/>

## ⚠️ Troubleshooting

If these tips do not solve your problem, please open a [discussion topic](https://github.com/openvinotoolkit/openvino_notebooks/discussions)
or create an [issue](https://github.com/openvinotoolkit/openvino_notebooks/issues)!

- To check some common installation problems, run `python check_install.py`. This script is located in the openvino_notebooks directory.
  Please run it after activating the `openvino_env` virtual environment.
- If you get an `ImportError`, doublecheck that you installed the Jupyter kernel. If necessary, choose the openvino\_env kernel from the _Kernel->Change Kernel_ menu) in Jupyter Lab or Jupyter Notebook
- If OpenVINO is installed globally, do not run installation commands in a terminal where setupvars.bat or setupvars.sh are sourced.
- For Windows installation, we recommend using _Command Prompt (cmd.exe)_, not _PowerShell_.

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)](#-contributors)
<div id='-contributors'/>

## 🧑‍💻 Contributors

<a href="https://github.com/openvinotoolkit/openvino_notebooks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openvinotoolkit/openvino_notebooks" />
</a>

Made with [contributors-img](https://contrib.rocks).

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-faq'/>

## ❓ FAQ

* [Which devices does OpenVINO support?](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html#doxid-openvino-docs-o-v-u-g-supported-plugins-supported-devices)
* [What is the first CPU generation you support with OpenVINO?](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html)
* [Are there any success stories about deploying real-world solutions with OpenVINO?](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html)


---

\* Other names and brands may be claimed as the property of others.
