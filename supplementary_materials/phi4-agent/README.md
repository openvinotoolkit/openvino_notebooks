# Phi-4-mini Agent on AI-PC with OpenVINO™ & 🤗 smolagents
In this notebook we will guide you on how to deploy a [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct) agent equipped with tools and MCP to run fully locally on your Intel<sup>®</sup> Core™ Ultra laptop with [OpenVINO™](https://docs.openvino.ai/2025/index.html) and [🤗 smolagents](https://github.com/huggingface/smolagents).

An LLM based agent is a program where the LLM control the entire workflow of the program. In this notebook we build a multi-step agent. Given a task, the LLM will decide which action to take in each step and when to terminate and return a final answer to the user.

We will demonstrate how the agent with the help of a few simple tools, like web-search and a code-completion tool (Also based on Phi-4-mini), the agent is able to code new tools and then use those tools summarize information into a PPT presentation for example.

Furthermore, we will demonstrate how with the use of a YouTube transcript MCP we can chat with YouTube videos without worrying about the prompt exploding to thousands of tokens and everything is done locally with speed.

## Prerequisites
Create a new Python environment for this notebook and activate it and make sure you have `git` installed. E.g.
```cmd
conda create -n phi4-demo python=3.11
conda activate phi4-demo
```

Then, lets start by installing all the packages we will need
```cmd
pip install -r requirements.txt
```

For this notebook we used a modified version of smolagents so we will need to clone the repository, apply our patch and install from source.
```
git clone https://github.com/huggingface/smolagents && cd smolagents
git checkout v1.14.0 && git apply ..\phi4_smolagents.patch
pip install .[mcp]
cd ..
```

> [!NOTE]
> To have the most up-to-date experience with OpenVINO you can also install the nightly versions:
> ```cmd
> pip install --pre -U openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
> pip install git+https://github.com/huggingface/optimum.git
> pip install git+https://github.com/huggingface/optimum-intel.git
> ```

## Prepare Phi-4-mini model for inference
We will use `optimum-cli` to convert and quantize Phi-4-mini model from the HuggingFace Hub to OpenVINO format
```cmd
optimum-cli export openvino --model microsoft/Phi-4-mini-instruct --task text-generation-with-past --weight-format int8 phi-4-mini-instruct-int8-ov
```

## Demo
Now we are ready to run our demo:
```cmd
python phi4_agent.py
```