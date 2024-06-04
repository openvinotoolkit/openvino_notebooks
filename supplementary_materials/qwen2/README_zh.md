简体中文 | [English](README.md)

# Qwen2.openvino Demo

这是如何使用 OpenVINO 部署 Qwen2 的示例

## 1. 环境配置

我们推荐您新建一个虚拟环境，然后按照以下安装依赖。
推荐在python3.10以上的环境下运行该示例。

Linux

```
python3 -m venv openvino_env

source openvino_env/bin/activate

python3 -m pip install --upgrade pip

pip install wheel setuptools

pip install -r requirements.txt
```

Windows Powershell

```
python3 -m venv openvino_env

.\openvino_env\Scripts\activate

python3 -m pip install --upgrade pip

pip install wheel setuptools

pip install -r requirements.txt
```

## 2. 转换模型

由于需要将Huggingface模型转换为OpenVINO IR模型，因此您需要下载模型并转换。

```
python3 convert.py --model_id qwen/Qwen2-7B-Instruct --precision int4 --output {your_path}/Qwen2-7B-Instruct-ov --modelscope
```

### 可以选择的参数

* `--model_id` - 用于从 Huggngface_hub (https://huggingface.co/models) 或 模型所在目录的路径（绝对路径）
* `--precision` - 模型精度：fp16, int8 或 int4。
* `--output` - 转换后模型保存的地址
* `--modelscope` - 通过魔搭社区下载模型

## 3. 运行流式聊天机器人

```
python3 chat.py --model_path {your_path}/Qwen2-7B-Instruct-ov --max_sequence_length 4096 --device CPU
```

### 可以选择的参数

* `--model_path` - OpenVINO IR 模型所在目录的路径。
* `--max_sequence_length` - 输出标记的最大大小。
* `--device` - 运行推理的设备。例如："CPU","GPU"。

## 例子

```
====Starting conversation====
用户: 你好
Qwen2-OpenVINO: 你好！有什么我可以帮助你的吗？

用户: 你是谁？
Qwen2-OpenVINO: 我是来自阿里云的超大规模语言模型，我叫通义千问。

用户: 请给我讲一个故事
Qwen2-OpenVINO: 好的，这是一个关于一只小兔子和它的朋友的故事。

有一天，小兔子和他的朋友们决定去森林里探险。他们带上食物、水和一些工具，开始了他们的旅程。在旅途中，他们遇到了各种各样的动物，包括松鼠、狐狸、小鸟等等。他们一起玩耍、分享食物，还互相帮助解决问题。最后，他们在森林的深处找到了一个神秘的洞穴，里面藏着许多宝藏。他们带着所有的宝藏回到了家，庆祝这次愉快的冒险。

用户: 请为这个故事起个标题
Qwen2-OpenVINO: "小兔子与朋友们的冒险之旅"
```

## 常见问题

1. 需要安装 OpenVINO C++ 推理引擎吗
   - 不需要

2. 一定要使用 Intel 的硬件吗？
   - 我们仅在 Intel 设备上尝试，我们推荐使用x86架构的英特尔设备，包括但不限制于：
   - 英特尔的CPU，包括个人电脑CPU 和服务器CPU。
   - 英特尔的集成显卡。 例如：Arc™，Iris® 系列。
   - 英特尔的独立显卡。例如：ARC™ A770 显卡。
  
3. 为什么OpenVINO没检测到我系统上的GPU设备？
   - 确保OpenCL驱动是安装正确的。
   - 确保你有足够的权限访问GPU设备
   - 更多信息可以参考[Install GPU drivers](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu#1-install-python-git-and-gpu-drivers-optional)

4. 是否支持C++？
   - C++示例可以[参考](https://github.com/openvinotoolkit/openvino.genai/tree/master/text_generation/causal_lm/cpp)
