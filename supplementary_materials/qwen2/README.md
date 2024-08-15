English | [简体中文](README_zh.md)

# Qwen2.openvino Demo

This sample shows how to deploy Qwen2 using OpenVINO

## 1. Environment configuration

We recommend that you create a new virtual environment and then install the dependencies as follows. The 
recommended Python version is `3.10+`.

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
> Note:
> If you are using an existing python environment, recommend following command to use all the dependencies with latest versions:  
> pip install -U --upgrade-strategy eager -r requirements.txt

## 2. Convert model

Since the Hugging Face model needs to be converted to an OpenVINO IR model, you need to download the model and convert.

```
python3 convert.py --model_id qwen/Qwen2-7B-Instruct --precision int4 --output {your_path}/Qwen2-7B-Instruct-ov --modelscope
```

### Parameters that can be selected

* `--model_id` - path (absolute path) to be used from Huggngface_hub (https://huggingface.co/models) or the directory
  where the model is located.
* `--precision` - model precision: fp16, int8 or int4.
* `--output` - the path where the converted model is saved.
* `--modelscope` - if downloading the model from Model Scope.

## 3. Run streaming chatbot

```
python3 chat.py --model_path {your_path}/Qwen2-7B-Instruct-ov --max_sequence_length 4096 --device CPU
```

### Parameters that can be selected

* `--model_path` - The path to the directory where the OpenVINO IR model is located.
* `--max_sequence_length` - Maximum size of output tokens.
* `--device` - The device to run inference on. e.g "CPU","GPU".

## Example

```
====Starting conversation====
User: hello
Qwen2-OpenVINO: Hello! How can I assist you today?

User: who are you ?
Qwen2-OpenVINO: I am an AI language model created by Alibaba Cloud. My purpose is to help users with their questions and provide them with accurate information. Is there anything specific you would like to know about me?

User: could you tell me a story ?
Qwen2-OpenVINO: Sure, here's a short story for you:

Once upon a time, in a small village nestled in the mountains, there lived a young girl named Lily who loved nature. She spent most of her days exploring the forest and watching the birds singing.

One day, while she was wandering through the woods, she stumbled upon a hidden cave deep within the forest. Inside, she found a beautiful crystal that sparkled with light. She picked it up and held it close to her heart, feeling a sense of joy and wonder.

As she walked away from the cave, she felt a sense of peace wash over her. She realized that sometimes, the things we miss the most are the simple things in life, like the beauty of nature or the warmth of the sun on our skin.

From that day forward, Lily made a habit of spending time in nature whenever she could. She would spend hours walking through the forest, watching the birds sing, and taking in the beauty around her. She knew that these moments were precious and that they would stay with her forever.

And so, Lily continued to live her life with a sense of joy and wonder, always cherishing the simple things in life.

User: please give this story a title
Qwen2-OpenVINO: "Nature's Magic: A Journey Through the Forest Crystal"
```

## FAQ

1. Do I need to install the OpenVINO C++ inference engine?
    - Unnecessary

2. Do I have to use Intel hardware?
    - It is recommended to use Intel x86 devices, and this is where we tested it. For example:
    - Intel CPU, including personal computer CPU and server CPU.
    - Intel's integrated GPU. For example: Arc™ Series and Iris® Series.
    - Intel's discrete graphics card. For example: ARC™ A770 graphics card.
  
3. Why can't OpenVINO find the GPU in my system(Linux)?
   - Ensure OpenCL drivers are installed correctly.
   - Ensure you enabled the right permissions for GPU device
   - More information can be found in [Install GPU drivers](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu#1-install-python-git-and-gpu-drivers-optional)

4. Is C++ supported ?
   - Please refer to this [example](https://github.com/openvinotoolkit/openvino.genai/tree/master/src)


Post your questions [here](https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/bd-p/distribution-openvino-toolkit).
