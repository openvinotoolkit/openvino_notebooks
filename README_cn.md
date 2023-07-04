[English](README.md) | 简体中文

<h1 align="center">📚 OpenVINO™ Notebooks</h1>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/LICENSE)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/treon_precommit.yml/badge.svg?event=push)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/treon_precommit.yml?query=event%3Apush)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml/badge.svg?event=push)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml?query=event%3Apush)

在这里，我们提供了一些可以运行的Jupyter* notebooks，用于学习和尝试使用OpenVINO™开发套件。这些notebooks旨在向各位开发者提供OpenVINO基础知识的介绍，并教会大家如何利用我们的API来优化深度学习推理。.

## 🚀 AI 趋势 - Notebooks 
查看最新notebooks代码示例，了解如何在英特尔CPU和GPU上优化和部署最近流行的深度学习模型。
| **Notebook** | **描述** | **预览** | **补充资料** |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [YOLOv8 - Optimization](notebooks/230-yolov8-optimization/)<br> | 利用 NNCF PTQ API 优化 YOLOv8 | <img src = "https://user-images.githubusercontent.com/29454499/212105105-f61c8aab-c1ff-40af-a33f-d0ed1fccc72e.png" width=300>  | [博客 - 如何用OpenVINO™让YOLOv8获得1000+ FPS性能？](https://mp.weixin.qq.com/s/PSfIZKp4PQtlLdwmn9Z6Bg) |
| [SAM - Segment Anything Model](notebooks/237-segment-anything/)<br>| 使用 Segment Anything以及OpenVINO™进行基于提示的对象分割掩码生成 | <img src=https://user-images.githubusercontent.com/29454499/231468849-1cd11e68-21e2-44ed-8088-b792ef50c32d.png width=300> | [博客 - AI分割一切——用OpenVINO™加速Meta SAM大模型 ](https://mp.weixin.qq.com/s/b7EVB6oouUKZGDCFbEi7Yw) |
| [ControlNet - Stable-Diffusion](notebooks/235-controlnet-stable-diffusion/)<br>| 利用ControlNet条件和OpenVINO™进行文本到图像生成 | <img src=https://user-images.githubusercontent.com/29454499/224541412-9d13443e-0e42-43f2-8210-aa31820c5b44.png width=300> | [Blog - Control your Stable Diffusion Model with ControlNet and OpenVINO](https://medium.com/@paularamos_5416/control-your-stable-diffusion-model-with-controlnet-and-openvino-f2aa7e6b1ebd)  |
| [Stable Diffusion v2](notebooks/236-stable-diffusion-v2/)<br>| 利用Stable Diffusion v2 以及 OpenVINO™进行文本到图像生成和无限缩放 |  <img src=https://user-images.githubusercontent.com/29454499/228882108-25c1f65d-4c23-4e1d-8ba4-f6164280a3e3.gif width=300>  | [博客 - AI作画升级，OpenVINO™ 和英特尔独立显卡助你快速生成视频](https://mp.weixin.qq.com/s/kfyTZK_Ybysceux6ChoLtA)  |
| [Whisper - Subtitles generation](notebooks/227-whisper-subtitles-generation/)<br> | 利用OpenAI Whisper以及OpenVINO™为视频生成字幕 | <img src=https://user-images.githubusercontent.com/29454499/204548693-1304ef33-c790-490d-8a8b-d5766acb6254.png  width=300>  |   |
| [CLIP - zero-shot-image-classification](notebooks/228-clip-zero-shot-image-classification)<br> | 利用CLIP 以及 OpenVINO™执行零样本图像分类 | <img src=https://user-images.githubusercontent.com/29454499/207795060-437b42f9-e801-4332-a91f-cc26471e5ba2.png width=300>  | [Blog - Generative AI and Explainable AI with OpenVINO ](https://medium.com/@paularamos_5416/generative-ai-and-explainable-ai-with-openvino-2b5f8e4e720b#:~:text=pix2pix%2Dimage%2Dediting-,Explainable%20AI%20with%20OpenVINO,-Explainable%20AI%20is)  |
| [BLIP - Visual-language-processing](notebooks/233-blip-visual-language-processing/)<br>| 利用BLIP以及OpenVINO™进行视觉问答和图像字幕 | <img src=https://user-images.githubusercontent.com/29454499/221933762-4ff32ecb-5e5d-4484-80e1-e9396cb3c511.png width=300>  | [Blog - Multimodality with OpenVINO — BLIP](https://medium.com/@paularamos_5416/multimodality-with-openvino-blip-b20bd3a2c87)  |
| [pix2pix - Image-editing](notebooks/231-instruct-pix2pix-image-editing/)<br>| 利用InstructPix2Pix进行图像编辑 | <<img src=https://user-images.githubusercontent.com/29454499/219943222-d46a2e2d-d348-4259-8431-37cf14727eda.png width=300>  | [Blog - Generative AI and Explainable AI with OpenVINO](https://medium.com/@paularamos_5416/generative-ai-and-explainable-ai-with-openvino-2b5f8e4e720b#:~:text=2.-,InstructPix2Pix,-Pix2Pix%20is%20a)  |

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()


## 目录

- [目录](#目录)
- [📝 安装指南](#-安装指南)
- [🚀 开始](#-开始)
	- [💻 第一步](#-第一步)
	- [⌚ 转换 \& 优化](#-转换--优化)
	- [🎯 模型演示](#-模型演示)
	- [🏃 模型训练](#-模型训练)
	- [📺 实时演示](#-实时演示)
- [⚙️ 系统要求](#️-系统要求)
- [⚙️ System Requirements](#️-system-requirements)
- [💻 运行Notebooks](#-运行notebooks)
	- [启动单个Notebook](#启动单个notebook)
	- [启动所有Notebooks](#启动所有notebooks)
- [🧹 清理](#-清理)
- [⚠️ 故障排除](#️-故障排除)
- [🧑‍💻 贡献者](#-贡献者)
- [❓ 常见问题解答](#-常见问题解答)

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-installation-guide'/>

## 📝 安装指南

OpenVINO Notebooks需要预装Python和Git， 针对不同操作系统的安装参考以下英语指南:

| [Windows](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) | [Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu) | [macOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS) | [Red Hat](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [CentOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [Azure ML](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML) | [Docker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Docker) | [Amazon SageMaker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/SageMaker)|
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |--------------------------------------------------------------------------- |
	
[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-getting-started'/>

## 🚀 开始

Jupyter notebooks 分为四个大类，选择一个跟你需求相关的开始试试吧。祝你好运！ 

<div id='-first-steps'/>

### 💻 第一步

演示如何使用OpenVINO的Python API进行推理的简短教程。

| [001-hello-world](notebooks/001-hello-world/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F001-hello-world%2F001-hello-world.ipynb) | [002-openvino-api](notebooks/002-openvino-api/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F002-openvino-api%2F002-openvino-api.ipynb) | [003-hello-segmentation](notebooks/003-hello-segmentation/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F003-hello-segmentation%2F003-hello-segmentation.ipynb) | [004-hello-detection](notebooks/004-hello-detection/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F004-hello-detection%2F004-hello-detection.ipynb) | 
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |  
|使用OpenVINO进行图像分类 | 学习使用OpenVINO Python API | 使用OpenVINO进行语义分割 | 使用OpenVINO进行文本检测  | 
| <img src="https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg" width=140> | <img src="https://user-images.githubusercontent.com/15709723/127787560-d8ec4d92-b4a0-411f-84aa-007e90faba98.png" width=250> | <img src="https://user-images.githubusercontent.com/15709723/128290691-e2eb875c-775e-4f4d-a2f4-15134044b4bb.png" width=150> | <img src="https://user-images.githubusercontent.com/36741649/128489933-bf215a3f-06fa-4918-8833-cb0bf9fb1cc7.jpg" width=150>  | 

<div id='-convert--optimize'/>

### ⌚ 转换 & 优化 

解释如何使用OpenVINO工具进行模型优化和量化的教程。


| Notebook | Description | 
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | 
| [101-tensorflow-classification-to-openvino](notebooks/101-tensorflow-classification-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F101-tensorflow-classification-to-openvino%2F101-tensorflow-classification-to-openvino.ipynb) | 转换 TensorFlow模型为OpenVINO IR | 
| [102-pytorch-onnx-to-openvino](notebooks/102-pytorch-onnx-to-openvino/) | 转换PyTorch模型为OpenVINO IR | 
| [103-paddle-to-openvino](notebooks/103-paddle-to-openvino/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F103-paddle-to-openvino%2F103-paddle-to-openvino-classification.ipynb) | 转换PaddlePaddle模型为OpenVINO IR | 
| [104-model-tools](notebooks/104-model-tools/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F104-model-tools%2F104-model-tools.ipynb) | 从Open Model Zoo进行模型下载，转换以及进行基线测试 | 
| [105-language-quantize-bert](notebooks/105-language-quantize-bert/) | 优化及量化BERT预训练模型 |
| [106-auto-device](notebooks/106-auto-device/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F106-auto-device%2F106-auto-device.ipynb) | 演示如何使用AUTO设备 |
| [107-speech-recognition-quantization](notebooks/107-speech-recognition-quantization/) | 优化及量化预训练Wav2Vec2语音模型 |
| [109-performance-tricks](notebooks/109-performance-tricks/)| OpenVINO™ 的优化技巧|
| [110-ct-segmentation-quantize](notebooks/110-ct-segmentation-quantize/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F110-ct-segmentation-quantize%2F110-ct-scan-live-inference.ipynb) | 量化肾脏分割模型并展示实时推理 | 
| [111-yolov5-quantization-migration](notebooks/111-yolov5-quantization-migration)<br> | 在神經網絡壓縮框架 (NNCF) 上遷移基於 YOLOv5 POT API 的量化管道 | <img src = "https://user-images.githubusercontent.com/44352144/177097174-cfe78939-e946-445e-9fce-d8897417ef8e.png"  width=225> |
| [112-pytorch-post-training-quantization-nncf](notebooks/112-pytorch-post-training-quantization-nncf/) | 利用神经网络压缩框架(NNCF)在后训练模式下来量化PyTorch模型(无需模型微调)| 
| [113-image-classification-quantization](notebooks/113-image-classification-quantization/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F113-image-classification-quantization%2F113-image-classification-quantization.ipynb) | 量化mobilenet图片分类模型 | 
| [115-async-api](notebooks/115-async-api/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F115-async-api%2F115-async-api.ipynb) | 使用异步执行改进数据流水线 |
| [116-sparsity-optimization](notebooks/116-sparsity-optimization/)| 提高稀疏Transformer模型的性能 |
| [117-model-server](notebooks/117-model-server/)| OpenVINO模型服务（OVMS）介绍 |
| [118-optimize-preprocessing](notebooks/118-optimize-preprocessing/)| 提升图片预处理性能 |

<div id='-model-demos'/>

### 🎯 模型演示

演示对特定模型的推理。


| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [201-vision-monodepth](notebooks/201-vision-monodepth/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F201-vision-monodepth%2F201-vision-monodepth.ipynb) | 利用图像和视频进行单目深度估计 | <img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif" width=250> |
| [202-vision-superresolution-image](notebooks/202-vision-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-image.ipynb) | 使用超分辨率模型放大原始图像 | <img src="https://user-images.githubusercontent.com/36741649/170005347-e4409f9e-ec34-416b-afdf-a9d8185929ca.jpg" width="70">→<img src="https://user-images.githubusercontent.com/36741649/170005347-e4409f9e-ec34-416b-afdf-a9d8185929ca.jpg" width="130"> |
| [202-vision-superresolution-video](notebooks/202-vision-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F202-vision-superresolution%2F202-vision-superresolution-video.ipynb) | 使用超分辨率模型将360p视频转换为1080p视频 | <img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width=80>→<img src="https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif" width="125"> |
| [203-meter-reader](notebooks/203-meter-reader/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F203-meter-reader%2F203-meter-reader.ipynb) | PaddlePaddle预训练模型读取工业表计数据 | <img src="https://user-images.githubusercontent.com/91237924/166135627-194405b0-6c25-4fd8-9ad1-83fb3a00a081.jpg" width=225> |
| [205-vision-background-removal](notebooks/205-vision-background-removal/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F205-vision-background-removal%2F205-vision-background-removal.ipynb) | 使用显著目标检测移除并替换图像中的背景 | <img src="https://user-images.githubusercontent.com/15709723/125184237-f4b6cd00-e1d0-11eb-8e3b-d92c9a728372.png" width=455> |
| [206-vision-paddlegan-anime](notebooks/206-vision-paddlegan-anime/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F206-vision-paddlegan-anime%2F206-vision-paddlegan-anime.ipynb) | 使用GAN把图片变为动画效果 | <img src="https://user-images.githubusercontent.com/15709723/127788059-1f069ae1-8705-4972-b50e-6314a6f36632.jpeg" width=100>→<img src="https://user-images.githubusercontent.com/15709723/125184441-b4584e80-e1d2-11eb-8964-d8131cd97409.png" width=100> |
| [207-vision-paddlegan-superresolution](notebooks/207-vision-paddlegan-superresolution/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F207-vision-paddlegan-superresolution%2F207-vision-paddlegan-superresolution.ipynb) | 使用PaddleGAN模型以超分辨率放大小图像| |
| [208-optical-character-recognition](notebooks/208-optical-character-recognition/)<br> | 使用文本识别resnet对图像上的文本进行注释 | <img src="https://user-images.githubusercontent.com/36741649/129315292-a37266dc-dfb2-4749-bca5-2ac9c1e93d64.jpg" width=225> |
| [209-handwritten-ocr](notebooks/209-handwritten-ocr/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F209-handwritten-ocr%2F209-handwritten-ocr.ipynb) | 手写体中文及日文OCR | <img width="425" alt="handwritten_simplified_chinese_test" src="https://user-images.githubusercontent.com/36741649/132660640-da2211ec-c389-450e-8980-32a75ed14abb.png"> <br> 的人不一了是他有为在责新中任自之我们 |
| [210-raft-optical-flow](notebooks/210-raft-optical-flow/)<br> | RAFT光流估计的OpenVNO推理 | <img src="https://github.com/openvinotoolkit/openvino_notebooks/assets/102195992/fbcc603f-f4fb-4b7c-b9c3-e6e6bf270557" width=225> |
| [211-speech-to-text](notebooks/211-speech-to-text/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F211-speech-to-text%2F211-speech-to-text.ipynb) | 运行语音转文本模型的推理 | <img src="https://user-images.githubusercontent.com/36741649/140987347-279de058-55d7-4772-b013-0f2b12deaa61.png" width=225>|
| [213-question-answering](notebooks/213-question-answering/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F213-question-answering%2F213-question-answering.ipynb) | 根据上下文回答问题 | <img src="https://user-images.githubusercontent.com/4547501/152571639-ace628b2-e3d2-433e-8c28-9a5546d76a86.gif" width=225> |
| [215-image-inpainting](notebooks/215-image-inpainting/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F215-image-inpainting%2F215-image-inpainting.ipynb)| 用绘画中的图像填充缺失像素 | <img src="https://user-images.githubusercontent.com/4547501/167121084-ec58fbdb-b269-4de2-9d4c-253c5b95de1e.png" width=225> |
| [217-vision-deblur](notebooks/217-vision-deblur/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/217-vision-deblur?labpath=notebooks%2F217-vision-deblur%2F217-vision-deblur.ipynb)| 使用DeblurGAN-v2去除图像模糊 | <img src="https://user-images.githubusercontent.com/41332813/158430181-05d07f42-cdb8-4b7a-b7dc-e7f7d9391877.png" width=225> |
| [218-vehicle-detection-and-recognition](notebooks/218-vehicle-detection-and-recognition/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F218-vehicle-detection-and-recognition%2F218-vehicle-detection-and-recognition.ipynb) | 利用OpenVINO及预训练模型检测和识别车辆及其属性 | <img src = "https://user-images.githubusercontent.com/47499836/163544861-fa2ad64b-77df-4c16-b065-79183e8ed964.png" width=225> |
| [219-knowledge-graphs-conve](notebooks/219-knowledge-graphs-conve/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F219-knowledge-graphs-conve%2F219-knowledge-graphs-conve.ipynb) | 使用OpenVINO优化知识图谱嵌入模型(ConvE) ||
| [221-machine-translation](notebooks/221-machine-translation)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F221-machine-translation%2F221-machine-translation.ipynb) | 从英语到德语的实时翻译 |  |
| [222-vision-image-colorization](notebooks/222-vision-image-colorization/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F222-vision-image-colorization%2F222-vision-image-colorization.ipynb) | 使用OpenVINO及预训练模型对黑白图像染色 | <img src = "https://user-images.githubusercontent.com/18904157/166343139-c6568e50-b856-4066-baef-5cdbd4e8bc18.png" width=225> |
| [223-gpt2-text-prediction](notebooks/223-text-prediction/)<br> |  使用GPT-2对输入序列执行文本预测 | <img src=https://user-images.githubusercontent.com/91228207/185105225-0f996b0b-0a3b-4486-872d-364ac6fab68b.png  width=225> |
| [224-3D-segmentation-point-clouds](notebooks/224-3D-segmentation-point-clouds/)<br> |使用OpenVINO处理点云数据并进行3D分割 | <img src = "https://user-images.githubusercontent.com/91237924/185752178-3882902c-907b-4614-b0e6-ea1de08bf3ef.png" width=225> |
| [225-stable-diffusion-text-to-image](notebooks/225-stable-diffusion-text-to-image)<br> | 用Stable Diffusion由文本生成图像 | <img src=https://user-images.githubusercontent.com/29454499/216524089-ed671fc7-a78b-42bf-aa96-9f7c791a9419.png width=225>|
| [226-yolov7-optimization](notebooks/226-yolov7-optimization/)<br> | 使用NNCF PTQ API优化YOLOv7 | <img src=https://raw.githubusercontent.com/WongKinYiu/yolov7/main/figure/horses_prediction.jpg  width=225> |
| [227-whisper-subtitles-generation](notebooks/227-whisper-subtitles-generation/)<br> | 利用OpenAI Whisper及OpenVINO为视频生成字幕 | <img src=https://user-images.githubusercontent.com/29454499/204548693-1304ef33-c790-490d-8a8b-d5766acb6254.png  width=225> |
| [228-clip-zero-shot-image-classification](notebooks/228-clip-zero-shot-image-classification)<br> | 利用CLIP及OpenVINO进行零样本图像分类 | <img src=https://user-images.githubusercontent.com/29454499/207795060-437b42f9-e801-4332-a91f-cc26471e5ba2.png  width=500> |
| [229-distilbert-sequence-classification](notebooks/229-distilbert-sequence-classification/)<br> | 利用OpenVINO进行句子分类 | <img src = "https://user-images.githubusercontent.com/95271966/206130638-d9847414-357a-4c79-9ca7-76f4ae5a6d7f.png" width=225> |
| [230-yolov8-optimization](notebooks/230-yolov8-optimization/)<br> |使用NNCF PTQ API优化YOLOv8 | <img src = "https://user-images.githubusercontent.com/29454499/212105105-f61c8aab-c1ff-40af-a33f-d0ed1fccc72e.png" width=225> |
|[231-instruct-pix2pix-image-editing](notebooks/231-instruct-pix2pix-image-editing/)<br>| 利用InstructPix2Pix进行图像编辑 | <img src=https://user-images.githubusercontent.com/29454499/219943222-d46a2e2d-d348-4259-8431-37cf14727eda.png width=225> |
|[232-clip-language-saliency-map](notebooks/232-clip-language-saliency-map/)<br>|  基于CLIP和OpenVINO™的视觉语言显著性检测 | <img src=https://user-images.githubusercontent.com/29454499/218967961-9858efd5-fff2-4eb0-bde9-60852f4b31cb.JPG width=225> | 
|[233-blip-visual-language-processing](notebooks/233-blip-visual-language-processing/)<br>| 基于BLIP和OpenVINO™的视觉问答与图片注释 | <img src=https://user-images.githubusercontent.com/29454499/221933762-4ff32ecb-5e5d-4484-80e1-e9396cb3c511.png width=225> |
|[234-encodec-audio-compression](notebooks/234-encodec-audio-compression/)<br>| # 基于EnCodec和OpenVINO™的音频压缩 | <img src=https://github.com/facebookresearch/encodec/raw/main/thumbnail.png width=225> |
|[235-controlnet-stable-diffusion](notebooks/235-controlnet-stable-diffusion/)<br>| # 使用ControlNet状态调节Stable Diffusion 实现文字生成图片 | <img src=https://user-images.githubusercontent.com/29454499/224541412-9d13443e-0e42-43f2-8210-aa31820c5b44.png width=225> |
|[236-stable-diffusion-v2](notebooks/236-stable-diffusion-v2/)<br>| 利用Stable Diffusion v2 以及 OpenVINO™进行文本到图像生成和无限缩放使用 | <img src=https://user-images.githubusercontent.com/29454499/228882108-25c1f65d-4c23-4e1d-8ba4-f6164280a3e3.gif width=225> |
|[237-segment-anything](notebooks/237-segment-anything/)<br>| 使用 Segment Anything以及OpenVINO™进行基于提示的对象分割掩码生成 | <img src=https://user-images.githubusercontent.com/29454499/231468849-1cd11e68-21e2-44ed-8088-b792ef50c32d.png width=225> |

<div id='-model-training'/>

### 🏃 模型训练

包含训练神经网络代码的教程。
| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [301-tensorflow-training-openvino](notebooks/301-tensorflow-training-openvino/) | 从TensorFlow训练花朵分类模型，然后转换为OpenVINO IR | <img src="https://user-images.githubusercontent.com/15709723/127779607-8fa34947-1c35-4260-8d04-981c41a2a2cc.png" width=390> |
| [301-tensorflow-training-openvino-pot](notebooks/301-tensorflow-training-openvino/) | 使用POT量化花朵模型 | |
| [302-pytorch-quantization-aware-training](notebooks/302-pytorch-quantization-aware-training/) | 使用神经网络压缩框架（NNCF）量化PyTorch模型 | |
| [305-tensorflow-quantization-aware-training](notebooks/305-tensorflow-quantization-aware-training/) | 使用神经网络压缩框架（NNCF）量化TensorFlow模型 | |

<div id='-live-demos'/>

### 📺 实时演示
在网络摄像头或视频文件上运行的实时推理演示。
	
	
| Notebook | Description | Preview |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [401-object-detection-webcam](notebooks/401-object-detection-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F401-object-detection-webcam%2F401-object-detection.ipynb) | 使用网络摄像头或视频文件进行目标检测  | <img src="https://user-images.githubusercontent.com/4547501/141471665-82b28c86-cf64-4bfe-98b3-c314658f2d96.gif" width=225> |
| [402-pose-estimation-webcam](notebooks/402-pose-estimation-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F402-pose-estimation-webcam%2F402-pose-estimation.ipynb) | 使用网络摄像头或视频文件进行人体姿态估计 | <img src="https://user-images.githubusercontent.com/4547501/138267961-41d754e7-59db-49f6-b700-63c3a636fad7.gif" width=225> |
| [403-action-recognition-webcam](notebooks/403-action-recognition-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F403-action-recognition-webcam%2F403-action-recognition-webcam.ipynb) | 使用网络摄像头或视频文件进行动作识别 | <img src="https://user-images.githubusercontent.com/10940214/151552326-642d6e49-f5a0-4fc1-bf14-ae3f457e1fec.gif" width=225> |
| [404-style-transfer-webcam](notebooks/404-style-transfer-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F404-style-transfer-webcam%2F404-style-transfer.ipynb) | 使用网络摄像头或视频文件进行样式变换 | <img src="https://user-images.githubusercontent.com/109281183/203772234-f17a0875-b068-43ef-9e77-403462fde1f5.gif" width=250> |
| [405-paddle-ocr-webcam](notebooks/405-paddle-ocr-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?labpath=notebooks%2F405-paddle-ocr-webcam%2F405-paddle-ocr-webcam.ipynb) | 使用网络摄像头或视频文件进行OCR | <img src="https://raw.githubusercontent.com/yoyowz/classification/master/images/paddleocr.gif" width=225> |
| [406-3D-pose-estimation-webcam](notebooks/406-3D-pose-estimation-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks.git/main?labpath=notebooks%2F406-3D-pose-estimation-webcam%2F406-3D-pose-estimation.ipynb) | 使用网络摄像头或视频文件进行3D人体姿态估计 | <img src = "https://user-images.githubusercontent.com/42672437/183292131-576cc05a-a724-472c-8dc9-f6bc092190bf.gif" width=225> |
| [407-person-tracking-webcam](notebooks/407-person-tracking-webcam/)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F407-person-tracking-webcam%2F407-person-tracking.ipynb) | 使用网络摄像头或视频文件进行人体跟踪 | <img src = "https://user-images.githubusercontent.com/91237924/210479548-b70dbbaa-5948-4e49-b48e-6cb6613226da.gif" width=225> |


如果你遇到了问题，请查看[故障排除](#-troubleshooting), [常见问题解答](#-faq) 或者创建一个GitHub [discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions)。

带有![binder logo](https://mybinder.org/badge_logo.svg) 按键的Notebooks可以在无需安装的情况下运行。[Binder](https://mybinder.org/) 是一项资源有限的免费在线服务。 如果享有获得最佳性能体验，请遵循[安装指南](#-installation-guide)在本地运行Notebooks。


[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-system-requirements'/>

## ⚙️ 系统要求
## ⚙️ System Requirements

这些notebooks可以运行在任何地方，包括你的笔记本电脑，云VM，或者一个Docker容器。下表列出了所支持的操作系统和Python版本。

| 支持的操作系统                                              | [Python Version (64-bit)](https://www.python.org/) |
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

## 💻 运行Notebooks

### 启动单个Notebook

如果你希望启动单个的notebook（如：Monodepth notebook），运行以下命令：

```bash
jupyter 201-vision-monodepth.ipynb
```

### 启动所有Notebooks

```bash
jupyter lab notebooks
```

在浏览器中，从Jupyter Lab侧边栏的文件浏览器中选择一个notebook文件，每个notebook文件都位于notebooks目录中的子目录中。

<img src="https://user-images.githubusercontent.com/15709723/120527271-006fd200-c38f-11eb-9935-2d36d50bab9f.gif">

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-cleaning-up'/>

## 🧹 清理

<div id='-shut-down-jupyter-kernel'>
&#8226; 停止Jupyter Kernel

按 Ctrl-c 结束 Jupyter session，会弹出一个提示框 Shutdown this Jupyter server (y/[n])? 输入 y 并按 回车。
</div>	
	
<div id='-deactivate-virtual-environment'>
&#8226; 注销虚拟环境

注销虚拟环境：只需在激活了 openvino_env 的终端窗口中运行 deactivate 即可。

重新激活环境：在Linux上运行 source openvino_env/bin/activate 或者在Windows上运行 openvino_env\Scripts\activate 即可，然后输入 jupyter lab 或 jupyter notebook 即可重新运行notebooks。
</div>	
	
<div id='-delete-virtual-environment'>
&#8226; >删除虚拟环境 _(可选)_

直接删除 openvino_env 目录即可删除虚拟环境：
</div>	
	
<div id='-on-linux-and-macos'>
&#8226; Linux和macOS:

```bash
rm -rf openvino_env
```
</div>	

<div id='-on-windows'>
&#8226; Windows:

```bash
rmdir /s openvino_env
```
</div>	

<div id='-remove-openvino-env-kernel'>
&#8226;从Jupyter中移除openvino_env Kernel

```bash
jupyter kernelspec remove openvino_env
```
</div>	

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-troubleshooting'/>

## ⚠️ 故障排除

如果以下方法无法解决您的问题，欢迎创建一个[discussion topic](https://github.com/openvinotoolkit/openvino_notebooks/discussions)
或[issue](https://github.com/openvinotoolkit/openvino_notebooks/issues)!

- 运行 python check_install.py 可以帮助检查一些常见的安装问题，该脚本位于openvino_notebooks 目录中。
  记得运行该脚本之前先激活 openvino_env 虚拟环境。
- 如果出现 ImportError ，请检查是否安装了 Jupyter Kernel。如需手动设置kernel，从 Jupyter Lab 或 Jupyter Notebook 的_Kernel->Change Kernel_菜单中选择openvino_env内核。
- 如果OpenVINO是全局安装的，不要在执行了setupvars.bat或setupvars.sh的终端中运行安装命令。
- 对于Windows系统，我们建议使用_Command Prompt (cmd.exe)，而不是_PowerShell。

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)](#-contributors)
<div id='-contributors'/>

## 🧑‍💻 贡献者

<a href="https://github.com/openvinotoolkit/openvino_notebooks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openvinotoolkit/openvino_notebooks" />
</a>

使用 [contributors-img](https://contrib.rocks)制作。

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-faq'/>

## ❓ 常见问题解答

* [OpenVINO支持哪些设备？](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html#doxid-openvino-docs-o-v-u-g-supported-plugins-supported-devices)
* [OpenVINO支持的第一代CPU是什么？](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html)
* [在使用OpenVINO部署现实世界解决方案方面有没有成功的案例？](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html)


---

\*其他名称和品牌可能被视为他人的财产。
