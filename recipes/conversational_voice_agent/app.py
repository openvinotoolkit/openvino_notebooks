import argparse
from pathlib import Path
from typing import Tuple, List

import gradio as gr
import numpy as np
from optimum.intel import OVModelForCausalLM
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from bark_utils import OVBark, SAMPLE_RATE

# todo: get chat template from transformers
CHAT_MODEL_TEMPLATES = {
    "llama2": {
        "system_configuration": "<<SYS>>\nYou're Adrishuo - a conversational agent. You talk to a customer. Your task is to recommend the customer products based on their needs.\n<</SYS>>\n\n",
        "message_template": "[INST] {} [/INST] {}\n"
    }
}

chat_model: OVModelForCausalLM = None
chat_tokenizer: PreTrainedTokenizer = None
system_configuration: str = None
message_template: str = None

tts_model: OVBark = None


def load_tts_model(model_dir: Path, speaker_type: str) -> None:
    """
        Load text-to-speech model and assign it to a global variable

        Params:
            model_dir: dir with the TTS model
            speaker_type: male or female
        """
    global tts_model

    text_encoder_path0 = model_dir / "text_encoder" / "bark_text_encoder_0.xml"
    text_encoder_path1 = model_dir / "text_encoder" / "bark_text_encoder_1.xml"
    coarse_encoder_path = model_dir / "coarse_model" / "bark_coarse_encoder.xml"
    fine_model_dir = model_dir / "fine_model"

    tts_model = OVBark(text_encoder_path0, text_encoder_path1, coarse_encoder_path, fine_model_dir, device="AUTO", speaker=speaker_type)


def load_chat_model(model_dir: Path) -> None:
    """
    Load chat model and assign it to a global variable

    Params:
        model_dir: dir with the chat model
    """
    global chat_model, chat_tokenizer, system_configuration, message_template

    model_name = model_dir.name.split("-")[0]
    # system and message templates specific to the loaded model
    system_configuration = CHAT_MODEL_TEMPLATES[model_name]["system_configuration"]
    message_template = CHAT_MODEL_TEMPLATES[model_name]["message_template"]

    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    chat_model = OVModelForCausalLM.from_pretrained(model_dir, device="AUTO", config=AutoConfig.from_pretrained(model_dir), ov_config=ov_config)
    chat_tokenizer = AutoTokenizer.from_pretrained(model_dir)


def respond(prompt: str) -> str:
    """
    Respond to the current prompt

    Params:
        prompt: user's prompt
    Returns:
        The chat's response
    """
    inputs = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = chat_model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9, top_k=50)
    token = outputs[0, input_length:]
    return chat_tokenizer.decode(token).split("</s>")[0]


def chat(history: List) -> List[Tuple[str, str]]:
    """
    Chat function. It generates response based on a prompt

    Params:
        history: history of the messages (conversation) so far
    Returns:
        History with the latest chat's response
    """
    user_prompt = history[-1][0]
    messages = [message_template.format(human if human is not None else "", ai if ai is not None else "") for human, ai in history]
    messages.append(message_template.format(user_prompt, ""))
    conversation = system_configuration + "\n".join(messages)

    response = respond(conversation)
    history[-1][1] = response

    return history


def synthesize(conversation: List[Tuple[str, str]]) -> Tuple[int, np.ndarray]:
    """
    Generate audio from text

    Params:
        conversation: conversation history with the chatbot
    Returns:
        Sample rate and generated audio in form of numpy array
    """
    prompt = conversation[-1][-1]
    return SAMPLE_RATE, tts_model.generate_audio(prompt)


def transcribe(audio: Tuple[int, np.ndarray], conversation: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Transcribe audio to text

    Params:
        audio: audio to transcribe text from
        conversation: conversation history with the chatbot
    Returns:
        Text in the next pair of messages
    """
    sample_rate, audio = audio
    text = "I want to buy a car"

    conversation.append((text, None))
    return conversation


def create_UI(initial_message: str) -> gr.Blocks:
    """
    Create web user interface

    Params:
        initial_message: message to start with
    Returns:
        Demo UI
    """
    with gr.Blocks(title="Talk to Adrishuo - a conversational voice agent") as demo:
        gr.Markdown("""
        # Talk to Adrishuo - a conversational voice agent
        
        Instructions for use:
        - record your question/comment using the first audio widget ("Your voice input")
        - wait for the chatbot to response ("Chatbot")
        - wait for the output voice in the last audio widget ("Chatbot voice output")
        """)
        with gr.Row():
            input_audio_ui = gr.Audio(sources=["microphone"], scale=5, label="Your voice input")
            submit_audio_btn = gr.Button("Submit", variant="primary", scale=1)

        chatbot_ui = gr.Chatbot(value=[(None, initial_message)], label="Chatbot")
        output_audio_ui = gr.Audio(autoplay=True, interactive=False, label="Chatbot voice output")

        # events
        submit_audio_btn.click(transcribe, inputs=[input_audio_ui, chatbot_ui], outputs=chatbot_ui).then(chat, chatbot_ui, chatbot_ui)
        # chatbot_ui.change(synthesize, inputs=chatbot_ui, outputs=output_audio_ui)
    return demo


def run(chat_model_dir: Path, tts_model_dir: Path, speaker_type: str, public_interface: bool = False) -> None:
    """
    Run the assistant application

    Params
        chat_model_dir: dir with the chat model
        tts_model_dir: dir with the text-to-speech model
        speaker_type: type of voice: male or female
        public_interface: whether UI should be available publicly
    """
    # load chat model
    load_chat_model(chat_model_dir)
    # load bark model
    load_tts_model(tts_model_dir, speaker_type)

    history = []
    initial_prompt = "Please introduce yourself and greet the customer"
    # get initial greeting
    history = chat(initial_prompt, history)
    initial_message = history[0][1]

    # create user interface
    demo = create_UI(initial_message)
    # launch demo
    demo.launch(share=public_interface)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat_model_dir', type=str, default="model/llama2-7B-INT8", help="Path to the chat model directory")
    parser.add_argument('--tts_model_dir', type=str, default="model/TTS_bark_small", help="Path to the text-to-speech model directory")
    parser.add_argument('--tts_speaker_type', type=str, default="male", choices=["male", "female"], help="The speaker's voice type")
    parser.add_argument('--public_interface', default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run(Path(args.chat_model_dir), Path(args.tts_model_dir), args.tts_speaker_type, args.public_interface)
