English | [简体中文](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README_cn.md)

<h1 align="center">📚 OpenVINO™ Notebooks</h1>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/LICENSE)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/nbval.yml/badge.svg)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/nbval.yml?query=branch%3Amain)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml/badge.svg)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/nbval.yml?query=branch%3Amain)

A collection of ready-to-run Jupyter notebooks for learning and experimenting with the OpenVINO™ Toolkit. The notebooks provide an introduction to OpenVINO basics and teach developers how to leverage our API for optimized deep learning inference.

**NOTE: The main branch of this repository was updated to support the new OpenVINO 2022.1 release.** To upgrade to the new release version, please run `pip install --upgrade -r requirements.txt` in your `openvino_env` virtual environment. If you need to install for the first time, see the [Installation Guide](#-installation-guide) section below. If you wish to use the previous Long Term Support (LTS) version of OpenVINO check out the [2021.4 branch](https://github.com/openvinotoolkit/openvino_notebooks/tree/2021.4). 

If you need help, please start a GitHub [Discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions).  

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

## Table of Contents

* [➤ 📝 Installation Guide](#-installation-guide)
	* [Windows](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows)
	* [Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu)
	* [macOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS)
	* [Red Hat](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS)
	* [CentOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS)
	* [Azure ML](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML)
	* [Docker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Docker)
* [➤ 🚀 Getting Started](#-getting-started)
	* [First steps with OpenVINO](#-first-steps)
	* [Convert & Optimize](#-convert--optimize)
	* [Model Demos](#-model-demos)
	* [Model Training](#-model-training)
	* [Live Demos](#-live-demos)
* [➤ ⚙️ System Requirements](#-system-requirements)
* [➤ 💻 Run the Notebooks](#-run-the-notebooks)
* [➤ 🧹 Cleaning Up](#-cleaning-up)
* [➤ ⚠️ Troubleshooting](#-troubleshooting)
* [➤ 🧑‍💻 Contributors](#-contributors)
* [➤ ❓ FAQ](#-faq)

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-installation-guide'/>

## 📝 Installation Guide

OpenVINO Notebooks require Python and Git. To get started, select the guide for your operating system or environment:

| [Windows](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) | [Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu) | [macOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS) | [Red Hat](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [CentOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [Azure ML](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML) | [Docker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Docker) |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
	
[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-getting-started'/>

## 🚀 Getting Started

The Jupyter notebooks are categorized into four classes, select one related to your needs or give them all a try. Good Luck! 

<div id='-first-steps'/>

### 💻 First steps

Brief tutorials that demonstrate how to use OpenVINO's Python API for inference.

| [001-hello-world](001-hello-world/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F001-hello-world%2F001-hello-world.ipynb) | [002-openvino-api](002-openvino-api/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F002-openvino-api%2F002-openvino-api.ipynb) | [003-hello-segmentation](003-hello-segmentation/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F003-hello-segmentation%2F003-hello-segmentation.ipynb) | [004-hello-detection](004-hello-detection/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F004-hello-detection%2F004-hello-detection.ipynb) | 
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |  
| Classify an image with OpenVINO | Learn the OpenVINO Python API | Semantic segmentation with OpenVINO | Text detection with OpenVINO  | 
| <img src="https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg" width=140> | <img src="https://user-images.githubusercontent.com/15709723/127787560-d8ec4d92-b4a0-411f-84aa-007e90faba98.png" width=250> | <img src="https://user-images.githubusercontent.com/15709723/128290691-e2eb875c-775e-4f4d-a2f4-15134044b4bb.png" width=150> | <img src="https://user-images.githubusercontent.com/36741649/128489933-bf215a3f-06fa-4918-8833-cb0bf9fb1cc7.jpg" width=150>  | 

<div id='-convert--optimize'/>

### ⌚ Convert & Optimize 

Tutorials that explain how to optimize and quantize models with OpenVINO tools.
	
| [101-tensorflow-to-openvino](101-tensorflow-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F101-tensorflow-to-openvino%2F101-tensorflow-to-openvino.ipynb) |  [102-pytorch-onnx-to-openvino](102-pytorch-onnx-to-openvino/) | [103-paddle-onnx-to-openvino](103-paddle-onnx-to-openvino-classification/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F103-paddle-onnx-to-openvino-classification%2F103-paddle-onnx-to-openvino-classification.ipynb) | [104-model-tools](104-model-tools/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F104-model-tools%2F104-model-tools.ipynb) | 
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |  
| Convert TensorFlow models to OpenVINO IR | Convert PyTorch models to OpenVINO IR | Convert PaddlePaddle models to OpenVINO IR | Download, convert and benchmark models from Open Model Zoo  | 
| <img src="https://user-images.githubusercontent.com/15709723/127779167-9d33dcc6-9001-4d74-a089-8248310092fe.png" width=250> | <img src="https://user-images.githubusercontent.com/15709723/127779246-32e7392b-2d72-4a7d-b871-e79e7bfdd2e9.png" width=300 > | <img src="https://user-images.githubusercontent.com/15709723/127779326-dc14653f-a960-4877-b529-86908a6f2a61.png" width=300>  | <img src="https://user-images.githubusercontent.com/10940214/157541917-c5455105-b0d9-4adf-91a7-fbc142918015.png" width=150>  |
	
More amazing notebooks here! 

<p>
<details>
<summary> Click here to show complete list!  </summary> 

| Notebook | Description | 
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | 
| [101-tensorflow-to-openvino](101-tensorflow-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F101-tensorflow-to-openvino%2F101-tensorflow-to-openvino.ipynb) | Convert TensorFlow models to OpenVINO IR | 
| [102-pytorch-onnx-to-openvino](102-pytorch-onnx-to-openvino/) | Convert PyTorch models to OpenVINO IR | 
| [103-paddle-onnx-to-openvino](103-paddle-onnx-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F103-paddle-onnx-to-openvino%2F103-paddle-onnx-to-openvino-classification.ipynb) | Convert PaddlePaddle models to OpenVINO IR | 
| [104-model-tools](104-model-tools/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F104-model-tools%2F104-model-tools.ipynb) | Download, convert and benchmark models from Open Model Zoo | 
| [105-language-quantize-bert](105-language-quantize-bert/) | Optimize and quantize a pre-trained BERT model |
| [106-auto-device](106-auto-device/) | Demonstrate how to use AUTO Device |
| [107-speech-recognition-quantization](107-speech-recognition-quantization/) | Optimize and quantize a pre-trained Wav2Vec2 speech model |
| [110-ct-segmentation-quantize](110-ct-segmentation-quantize/)<br> | Quantize a kidney segmentation model and show live inference | 
| [111-detection-quantization](111-detection-quantization/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F111-detection-quantization%2F111-detection-quantization.ipynb) | Quantize an object detection model | 
| [112-pytorch-post-training-quantization-nncf](112-pytorch-post-training-quantization-nncf/) | Use Neural Network Compression Framework (NNCF) to quantize PyTorch model in post-training mode (without model fine-tuning)| 
| [113-image-classification-quantization](113-image-classification-quantization/) | Image Classification Models with POT |
| [114-quantization-simplified-mode](114-quantization-simplified-mode/) | Quantize Image Classification Models with POT in Simplified Mode|
| [115-async-api](115-async-api/) | Use Asynchronous Execution to Improve Data Pipelining| |
</details>
</p>

<div id='-model-demos'/>

### 🎯 Model Demos

Demos that demonstrate inference on a particular model.
	
| [210-ct-scan-live-inference](210-ct-scan-live-inference/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F210-ct-scan-live-inference%2F210-ct-scan-live-inference.ipynb) | [211-speech-to-text](211-speech-to-text/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F211-speech-to-text%2F211-speech-to-text.ipynb) | [213-question-answering](213-question-answering/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F213-question-answering%2F213-question-answering.ipynb) | [208-optical-character-recognition](208-optical-character-recognition/)<br> |  [209-handwritten-ocr](209-handwritten-ocr/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F209-handwritten-ocr%2F209-handwritten-ocr.ipynb) |  
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | 
| Show live inference on segmentation of CT-scan data | Run inference on speech-to-text recognition model | Answer your questions basing on a context | Annotate text on images using text recognition resnet | OCR for handwritten simplified Chinese and Japanese |
|<img src="https://user-images.githubusercontent.com/15709723/134784204-cf8f7800-b84c-47f5-a1d8-25a9afab88f8.gif" width=225>| <img src="https://user-images.githubusercontent.com/36741649/140987347-279de058-55d7-4772-b013-0f2b12deaa61.png" width=225> | <img src="https://user-images.githubusercontent.com/4547501/152571639-ace628b2-e3d2-433e-8c28-9a5546d76a86.gif" width=225> | <img src="https://user-images.githubusercontent.com/36741649/129315292-a37266dc-dfb2-4749-bca5-2ac9c1e93d64.jpg" width=225> | <img width="425" alt="handwritten_simplified_chinese_test" src="https://user-images.githubusercontent.com/36741649/132660640-da2211ec-c389-450e-8980-32a75ed14abb.png"> <br> 的人不一了是他有为在责新中任自之我们 |
	
More amazing notebooks here! 

<p>
<details>
<summary> Click here to show complete list! </summary>
	
	
| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [201-vision-monodepth](201-vision-monodepth/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F201-vision-monodepth%2F201-vision-monodepth.ipynb) | Monocular depth estimation with images and video | <img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif" width=250> |
| [202-vision-superresolution-image](202-vision-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-image.ipynb) | Upscale raw images with a super resolution model | <img src="https://user-images.githubusercontent.com/36741649/170005347-e4409f9e-ec34-416b-afdf-a9d8185929ca.jpg" width="70">→<img src="https://user-images.githubusercontent.com/36741649/170005347-e4409f9e-ec34-416b-afdf-a9d8185929ca.jpg" width="130"> |
| [202-vision-superresolution-video](202-vision-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-video.ipynb) | Turn 360p into 1080p video using a super resolution model | <img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width=80>→<img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width="125"> |
| [203-meter-reader](203-meter-reader/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F203-meter-reader%2F203-meter-reader.ipynb) | PaddlePaddle pre-trained models to read industrial meter's value | <img src="https://user-images.githubusercontent.com/91237924/166135627-194405b0-6c25-4fd8-9ad1-83fb3a00a081.jpg" width=225> |
| [204-named-entity-recognition](204-named-entity-recognition/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F204-named-entity-recognition%2F204-named-entity-recognition.ipynb) | Perform named entity recognition on simple text | <img src="https://user-images.githubusercontent.com/33627846/169470030-0370963e-6ad8-49e3-be7a-f02a2c677733.gif" width="225"> |
| [205-vision-background-removal](205-vision-background-removal/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F205-vision-background-removal%2F205-vision-background-removal.ipynb) | Remove and replace the background in an image using salient object detection | <img src="https://user-images.githubusercontent.com/15709723/125184237-f4b6cd00-e1d0-11eb-8e3b-d92c9a728372.png" width=455> |
| [206-vision-paddlegan-anime](206-vision-paddlegan-anime/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F206-vision-paddlegan-anime%2F206-vision-paddlegan-anime.ipynb) | Turn an image into anime using a GAN | <img src="https://user-images.githubusercontent.com/15709723/127788059-1f069ae1-8705-4972-b50e-6314a6f36632.jpeg" width=100>→<img src="https://user-images.githubusercontent.com/15709723/125184441-b4584e80-e1d2-11eb-8964-d8131cd97409.png" width=100> |
| [207-vision-paddlegan-superresolution](207-vision-paddlegan-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F207-vision-paddlegan-superresolution%2F207-vision-paddlegan-superresolution.ipynb) | Upscale small images with superresolution using a PaddleGAN model| |
| [208-optical-character-recognition](208-optical-character-recognition/)<br> | Annotate text on images using text recognition resnet | <img src="https://user-images.githubusercontent.com/36741649/129315292-a37266dc-dfb2-4749-bca5-2ac9c1e93d64.jpg" width=225> |
| [209-handwritten-ocr](209-handwritten-ocr/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F209-handwritten-ocr%2F209-handwritten-ocr.ipynb) | OCR for handwritten simplified Chinese and Japanese | <img width="425" alt="handwritten_simplified_chinese_test" src="https://user-images.githubusercontent.com/36741649/132660640-da2211ec-c389-450e-8980-32a75ed14abb.png"> <br> 的人不一了是他有为在责新中任自之我们 |
| [210-ct-scan-live-inference](210-ct-scan-live-inference/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F210-ct-scan-live-inference%2F210-ct-scan-live-inference.ipynb) | Show live inference on segmentation of CT-scan data | <img src="https://user-images.githubusercontent.com/77325899/154280563-0e94f972-2d1a-44f9-a894-1b61699d1781.gif" width=225> |
| [211-speech-to-text](211-speech-to-text/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F211-speech-to-text%2F211-speech-to-text.ipynb) | Run inference on speech-to-text recognition model | <img src="https://user-images.githubusercontent.com/36741649/140987347-279de058-55d7-4772-b013-0f2b12deaa61.png" width=225>|
| [212-onnx-style-transfer](212-onnx-style-transfer/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F212-onnx-style-transfer%2F212-onnx-style-transfer.ipynb) | Transform images to five different styles with neural style transfer | <img src="https://user-images.githubusercontent.com/77325899/147358090-ff5b21f5-0efb-4aff-8444-9d07add49b92.png" width=100>→<img src="https://user-images.githubusercontent.com/77325899/147358009-0cf10d51-3150-40cb-a776-074558b98da5.png" width=100>|
| [213-question-answering](213-question-answering/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F213-question-answering%2F213-question-answering.ipynb) | Answer your questions basing on a context | <img src="https://user-images.githubusercontent.com/4547501/152571639-ace628b2-e3d2-433e-8c28-9a5546d76a86.gif" width=225> |
| [214-vision-paddle-classification](214-vision-paddle-classification/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F214-vision-paddle-classification%2F214-vision-paddle-classification.ipynb) | PaddlePaddle Image Classification with OpenVINO| |
| [215-image-inpainting](215-image-inpainting/)<br> | Fill missing pixels with image in-painting | <img src="https://user-images.githubusercontent.com/4547501/167121084-ec58fbdb-b269-4de2-9d4c-253c5b95de1e.png" width=225> |
| [216-license-plate-recognition](216-license-plate-recognition/)<br> | Recognize Chinese license plates in traffic | <img src="https://user-images.githubusercontent.com/70456146/162759539-4a0a996f-dabe-40ea-98d6-85b4dce8511d.png" width=225> |
| [217-vision-deblur](217-vision-deblur/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/217-vision-deblur?labpath=notebooks%2F217-vision-deblur%2F217-vision-deblur.ipynb)| Deblur Images with DeblurGAN-v2 | <img src="https://user-images.githubusercontent.com/41332813/158430181-05d07f42-cdb8-4b7a-b7dc-e7f7d9391877.png" width=225> |
| [218-vehicle-detection-and-recognition](218-vehicle-detection-and-recognition/)<br> | Use pre-trained models to detect and recognize vehicles and their attributes with OpenVINO | <img src = "https://user-images.githubusercontent.com/47499836/163544861-fa2ad64b-77df-4c16-b065-79183e8ed964.png" width=225> |
| [219-knowledge-graphs-conve](219-knowledge-graphs-conve/)<br> | Optimize the knowledge graph embeddings model (ConvE) with OpenVINO ||
| [220-yolov5-accuracy-check-and-quantization](220-yolov5-accuracy-check-and-quantization/)<br> | Quantize the Ultralytics YOLOv5 model and check accuracy using the OpenVINO POT API | <img src = "https://user-images.githubusercontent.com/44352144/177097174-cfe78939-e946-445e-9fce-d8897417ef8e.png" width=225> |
| [221-machine-translation](221-machine-translation)<br> | Real-time translation from English to German |  |
| [222-vision-image-colorization](222-vision-image-colorization/)<br> | Use pre-trained models to colorize black \& white images using OpenVINO | <img src = "https://user-images.githubusercontent.com/18904157/166343139-c6568e50-b856-4066-baef-5cdbd4e8bc18.png" width=225> 


</details>
</p>

<div id='-model-training'/>

### 🏃 Model Training

Tutorials that include code to train neural networks.
| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [301-tensorflow-training-openvino](301-tensorflow-training-openvino/) | Train a flower classification model from TensorFlow, then convert to OpenVINO IR | <img src="https://user-images.githubusercontent.com/15709723/127779607-8fa34947-1c35-4260-8d04-981c41a2a2cc.png" width=390> |
| [301-tensorflow-training-openvino-pot](301-tensorflow-training-openvino/) | Use Post-training Optimization Tool (POT) to quantize the flowers model | |
| [302-pytorch-quantization-aware-training](302-pytorch-quantization-aware-training/) | Use Neural Network Compression Framework (NNCF) to quantize PyTorch model | |
| [305-tensorflow-quantization-aware-training](305-tensorflow-quantization-aware-training/) | Use Neural Network Compression Framework (NNCF) to quantize TensorFlow model | |

<div id='-live-demos'/>

### 📺 Live Demos
Live inference demos that run on a webcam or video files.
	
| [401-object-detection-webcam](401-object-detection-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F401-object-detection-webcam%2F401-object-detection.ipynb) | [402-pose-estimation-webcam](402-pose-estimation-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F402-pose-estimation-webcam%2F402-pose-estimation.ipynb) | [403-action-recognition-webcam](403-action-recognition-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F403-action-recognition-webcam%2F403-action-recognition-webcam.ipynb) | [405-paddle-ocr-webcam](405-paddle-ocr-webcam/) |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------- | 
|Object detection with a webcam or video file | Human pose estimation with a webcam or video file |  Human action recognition with a webcam or video file |  OCR with a webcam or video file |
| <img src="https://user-images.githubusercontent.com/4547501/141471665-82b28c86-cf64-4bfe-98b3-c314658f2d96.gif" width=225> | <img src="https://user-images.githubusercontent.com/4547501/138267961-41d754e7-59db-49f6-b700-63c3a636fad7.gif" width=225> |  <img src="https://user-images.githubusercontent.com/10940214/151552326-642d6e49-f5a0-4fc1-bf14-ae3f457e1fec.gif" width=225> |   <img src="https://raw.githubusercontent.com/yoyowz/classification/master/images/ezgif.com-gif-maker.gif" width=225> |



If you run into issues, please check the [troubleshooting section](#-troubleshooting), [FAQs](#-faq) or start a GitHub [discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions). 

Notebooks with a ![binder logo](https://mybinder.org/badge_logo.svg) button can be run without installing anything. [Binder](https://mybinder.org/) is a free online service with limited resources. For the best performance, please follow the [Installation Guide](#-installation-guide) and run the notebooks locally.

You will have a lot of fun with this section:

| [Vision-monodepth](201-vision-monodepth/) | [CT-scan-live-inference](210-ct-scan-live-inference/) | [Object-detection-webcam](401-object-detection-webcam/) | [Pose-estimation-webcam](402-pose-estimation-webcam/) | [Action-recognition-webcam](403-action-recognition-webcam/) | 
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | 
| <img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif" width=250> | <img src="https://user-images.githubusercontent.com/15709723/134784204-cf8f7800-b84c-47f5-a1d8-25a9afab88f8.gif" width=225> | <img src="https://user-images.githubusercontent.com/4547501/141471665-82b28c86-cf64-4bfe-98b3-c314658f2d96.gif" width=225> | <img src="https://user-images.githubusercontent.com/4547501/138267961-41d754e7-59db-49f6-b700-63c3a636fad7.gif" width=225>  | <img src="https://user-images.githubusercontent.com/10940214/151552326-642d6e49-f5a0-4fc1-bf14-ae3f457e1fec.gif" width=225> | 


[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-system-requirements'/>

## ⚙️ System Requirements

The notebooks run almost anywhere &mdash; your laptop, a cloud VM, or even a Docker container. The table below lists the supported operating systems and Python versions. **Note:** Python 3.10 is not supported yet.

| Supported Operating System                                 | [Python Version (64-bit)](https://www.python.org/) |
| :--------------------------------------------------------- | :------------------------------------------------- |
| Ubuntu 18.04 LTS, 64-bit                                 | 3.6, 3.7, 3.8, 3.9                                      |
| Ubuntu 20.04 LTS, 64-bit                                 | 3.6, 3.7, 3.8, 3.9                                      |
| Red Hat Enterprise Linux 8, 64-bit                       | 3.6, 3.8, 3.9                                           |
| CentOS 7, 64-bit                                         | 3.6, 3.7, 3.8, 3.9                                      |
| macOS 10.15.x versions                                   | 3.6, 3.7, 3.8, 3.9                                      |
| Windows 10, 64-bit Pro, Enterprise or Education editions | 3.6, 3.7, 3.8, 3.9                                      |
| Windows Server 2016 or higher                            | 3.6, 3.7, 3.8, 3.9                                      |

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

* [Which devices does OpenVINO support?](https://docs.openvino.ai/2022.1/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html)
* [What is the first CPU generation you support with OpenVINO?](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html)
* [Are there any success stories about deploying real-world solutions with OpenVINO?](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html)


---

\* Other names and brands may be claimed as the property of others.
