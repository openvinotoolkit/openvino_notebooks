# Conversational Voice Agent with OpenVINO™

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/LICENSE)

The Conversational Voice Agent utilizes the OpenVINO™ toolkit to create a streamlined, voice-activated interface that developers can easily integrate and deploy. At its core, the application harnesses state-of-the-art models for speech recognition, natural language understanding, and speech synthesis. It's configured to understand user prompts, engage in dialogue, and provide auditory responses, facilitating an interactive and user-friendly conversational agent.

## Getting Started

Now, let's dive into the steps starting with installing Python. We recommend using Ubuntu to set up and run this project.

### Installing Prerequisites

This project requires Python 3.8 or higher and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

```shell
sudo apt install git gcc python3-venv python3-dev
```

_NOTE: If you are using Windows, you may need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) also._

### Setting up your Environment

#### Cloning the Repository

To clone the repository, run the following command:

```shell
git clone -b recipes https://github.com/openvinotoolkit/openvino_notebooks.git openvino_notebooks
```

The above will clone the repository into a directory named "openvino_notebooks" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_notebooks/recipes/conversational_voice_agent
```

#### Creating a Virtual Environment

To create a virtual environment, open your terminal or command prompt and navigate to the directory where you want to create the environment. Then, run the following command:

```shell
python3 -m venv venv
```
This will create a new virtual environment named "venv" in the current directory.

#### Activating the Environment

Activate the virtual environment using the following command:

```shell
source venv/bin/activate   # For Unix-based operating system such as Linux or macOS
```

_NOTE: If you are using Windows, use `venv\Scripts\activate` command instead._

This will activate the virtual environment and change your shell's prompt to indicate that you are now working within that environment.

#### Installing the Packages

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```
### How to Access LlaMA 2

_NOTE: If you already have access to the LlaMA 2 model weights, skip to the authentication step, which is mandatory for converting the LlaMA 2 model._

#### Accessing Original Weights from Meta AI

To access the original LlaMA 2 model weights:

Visit [Meta AI's website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and fill in your details, including your name, email, and organization.
Accept the terms and submit the form. You will receive an email granting access to download the model weights.
Using LlaMA 2 with Hugging Face
Set Up a Hugging Face Account: If you don't have one, create a [Hugging Face account](https://huggingface.co/welcome).

Authenticate with Meta AI: Go to the LlaMA 2 model page on Hugging Face. You'll need to enter the same email address you used for the Meta AI website to be authenticated. After authentication, you'll gain access to the model.

To use the model, authenticate using the Hugging Face CLI:

```shell
huggingface-cli login
```
When prompted to add the token as a git credential, respond with 'n'. This step ensures that you are logged into the Hugging Face API and ready to download the model.

Now, you're ready to download and optimize the models required to run the application.

## Model Conversion and Optimization

The application uses three separate models for its operation, each requiring conversion and optimization for use with OpenVINO™. Follow the order below to convert and optimize each model:

1. Automated Speech Recognition Distil-Whisper Conversion:
```shell
python convert_and_optimize_asr.py
```
This script will convert and optimize the automatic speech recognition (ASR) model.

2. Chat LLama2 Conversion:
```shell
python convert_and_optimize_chat.py --chat_model_type llama2-7B --quantize_weights int8
```
This script will handle the conversion and optimization of the chat model performing weights quantization. 

3. Text to Speech SpeechT5 Model Loading:

SpeechT5 model will be loaded.

```shell
python load_tts.py
```

After running the conversion scripts, you can run app.py to launch the application.

_NOTE: Running the above script may take up to 120 minutes (depending on your hardware Internet connection), as some models are huge (especially chatbots)._

## Running the Application (Gradio Interface)

Execute the `app.py` script with the following command, including all necessary model directory arguments:
```shell
python app.py --asr_model_dir path/to/asr_model --chat_model_dir path/to/chat_model --public_interface
```
Make sure to replace `path/to/asr_model`, `path/to/chat_model` with the actual paths to your respective models. Add `--public_interface` to make it publicly accessible.

### Accessing the Web Interface
Upon successful execution of the script, Gradio will provide a local URL, typically `http://127.0.0.1:XXXX`, which you can open in your web browser to start interacting with the voice agent. If you configured the application to be accessible publicly, Gradio will also provide a public URL.

Trying Out the Application
1. Navigate to the provided Gradio URL in your web browser.
2. You will see the Gradio interface with options to input voice.
3. To interact using voice:
    - Click on the microphone icon and speak your query.
    - Wait for the voice agent to process your speech and respond.
4. The voice agent will respond to your query in text and with synthesized speech.

Feel free to engage with the Conversational Voice Agent, ask questions, or give commands as per the agent's capabilities. This hands-on experience will help you better understand the agent's interactive quality and performance.

Enjoy exploring the capabilities of your Conversational Voice Agent!

## Troubleshooting and Resources
- Open a [discussion topic](https://github.com/openvinotoolkit/openvino_notebooks/discussions)
- Create an [issue](https://github.com/openvinotoolkit/openvino_notebooks/issues)
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2023.0/home.html)