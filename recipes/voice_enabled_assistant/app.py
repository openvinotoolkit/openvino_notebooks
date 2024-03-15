import argparse
import time
from pathlib import Path
from threading import Thread
from typing import Tuple, List, Optional

import gradio as gr
import librosa
import numpy as np
from optimum.intel import OVModelForCausalLM, OVModelForSpeechSeq2Seq
from transformers import AutoConfig, AutoTokenizer, AutoProcessor, PreTrainedTokenizer, TextIteratorStreamer
from transformers.generation.streamers import BaseStreamer

# Global variables initialization
AUDIO_WIDGET_SAMPLE_RATE = 16000
SYSTEM_CONFIGURATION = ("You're Adrishuo - a helpful, respectful and honest doctor assistant. Your role is talking to a patient, who just came in. "
                        "Your task is to get all symptoms from the patient and summarize them to the doctor. You cannot treat the patient yourself.")
GREET_THE_CUSTOMER = "Please introduce yourself and greet the patient"
SUMMARIZE_THE_CUSTOMER = "Summarize the above patient to the doctor. Use only information provided by patient."
NEURAL_CHAT_MODEL_TEMPLATE = ("{% if messages[0]['role'] == 'system' %}"
                              "{% set loop_messages = messages[1:] %}"
                              "{% set system_message = messages[0]['content'] %}"
                              "{% else %}"
                              "{% set loop_messages = messages %}"
                              "{% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
                              "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. "
                              "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'t know the answer to a question, please don\\'t share false information.' %}"
                              "{% endif %}"
                              "{{ '### System:\\n' + system_message.strip() + '\\n' }}"
                              "{% for message in loop_messages %}"
                              "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
                              "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
                              "{% endif %}"
                              "{% set content = message['content'] %}"
                              "{% if message['role'] == 'user' %}"
                              "{{ '### User:\\n' + content.strip() + '\\n' }}"
                              "{% elif message['role'] == 'assistant' %}"
                              "{{ '### Assistant:\\n' + content.strip() + '\\n'}}"
                              "{% endif %}"
                              "{% endfor %}"
                              )

# Initialize Model variables
chat_model: Optional[OVModelForCausalLM] = None
chat_tokenizer: Optional[PreTrainedTokenizer] = None
message_template: Optional[str] = None
asr_model: Optional[OVModelForSpeechSeq2Seq] = None
asr_processor: Optional[AutoProcessor] = None


def load_asr_model(model_dir: Path) -> None:
    """
    Load automatic speech recognition model and assign it to a global variable

    Params:
        model_dir: dir with the ASR model
    """
    global asr_model, asr_processor

    # create a distil-whisper model and its processor
    asr_model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, device="AUTO")
    asr_processor = AutoProcessor.from_pretrained(model_dir)


def load_chat_model(model_dir: Path) -> None:
    """
    Load chat model and assign it to a global variable

    Params:
        model_dir: dir with the chat model
    """
    global chat_model, chat_tokenizer, message_template

    # load llama model and its tokenizer
    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    chat_model = OVModelForCausalLM.from_pretrained(model_dir, device="AUTO", config=AutoConfig.from_pretrained(model_dir), ov_config=ov_config)
    chat_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # neural chat requires different template than specified in the tokenizer
    message_template = NEURAL_CHAT_MODEL_TEMPLATE if ("neural-chat" in model_dir.name) else chat_tokenizer.default_chat_template


def respond(prompt: str, streamer: BaseStreamer | None = None) -> str:
    """
    Respond to the current prompt

    Params:
        prompt: user's prompt
        streamer: if not None will use it to stream tokens
    Returns:
        The chat's response
    """
    start_time = time.time()  # Start time
    # tokenize input text
    inputs = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device)
    input_length = inputs.input_ids.shape[1]
    # generate response tokens
    outputs = chat_model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9, top_k=50, streamer=streamer)
    token = outputs[0, input_length:]
    # decode tokens into text
    end_time = time.time()  # End time
    print("Chat model response time: {:.2f} seconds".format(end_time - start_time))
    return chat_tokenizer.decode(token).split("</s>")[0]


def get_conversation(history: List[List[str]]) -> str:
    """
    Combines all messages into one string

    Params:
        history: history of the messages (conversation) so far
    Returns:
        All messages combined into one string
    """
    # the conversation must be in that format to use chat template
    conversation = [
        {"role": "system", "content": SYSTEM_CONFIGURATION},
        {"role": "user", "content": GREET_THE_CUSTOMER}
    ]
    # add prompts to the conversation
    for user_prompt, assistant_response in history:
        if user_prompt:
            conversation.append({"role": "user", "content": user_prompt})
        if assistant_response:
            conversation.append({"role": "assistant", "content": assistant_response})

    # use a template specific to the model
    return chat_tokenizer.apply_chat_template(conversation, chat_template=message_template, tokenize=False)


def generate_initial_greeting() -> str:
    """
    Generates customer/patient greeting

    Returns:
        Generated greeting
    """
    conv = get_conversation([[None, None]])
    return respond(conv)


def chat(history: List[List[str]]) -> List[List[str]]:
    """
    Chat function. It generates response based on a prompt

    Params:
        history: history of the messages (conversation) so far
    Returns:
        History with the latest chat's response (yields partial response)
    """
    # convert list of message to conversation string
    conversation = get_conversation(history)

    # use streamer to show response word by word
    chat_streamer = TextIteratorStreamer(chat_tokenizer, skip_prompt=True, Timeout=5)

    # generate response for the conversation in a new thread to deliver response token by token
    thread = Thread(target=respond, args=[conversation, chat_streamer])
    thread.start()

    # get token by token and merge to the final response
    history[-1][1] = ""
    for partial_text in chat_streamer:
        history[-1][1] += partial_text
        # "return" partial response
        yield history

    # wait for the thread
    thread.join()


def transcribe(audio: Tuple[int, np.ndarray], conversation: List[List[str]]) -> List[List[str]]:
    """
    Transcribe audio to text

    Params:
        audio: audio to transcribe text from
        conversation: conversation history with the chatbot
    Returns:
        User prompt as a text
    """
    start_time = time.time()  # Start time for ASR process

    sample_rate, audio = audio
    # the whisper model requires 16000Hz, not 44100Hz
    audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=AUDIO_WIDGET_SAMPLE_RATE).astype(np.int16)

    # get input features from the audio
    input_features = asr_processor(audio, sampling_rate=AUDIO_WIDGET_SAMPLE_RATE, return_tensors="pt").input_features
    # get output
    predicted_ids = asr_model.generate(input_features)
    # decode output to text
    transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    end_time = time.time()  # End time for ASR process
    print(f"ASR model response time: {end_time - start_time:.2f} seconds")  # Print the ASR processing time

    # add the text to the conversation
    conversation.append([transcription, None])
    return conversation


def summarize(conversation: List) -> str:
    """
    Summarize the patient case

    Params
        conversation: history of the messages so far
    Returns:
        Summary
    """
    conversation.append([SUMMARIZE_THE_CUSTOMER, None])
    for partial_summary in chat(conversation):
        yield partial_summary[-1][1].split("</s>")[0]


def create_UI(initial_message: str) -> gr.Blocks:
    """
    Create web user interface

    Params:
        initial_message: message to start with
    Returns:
        Demo UI
    """
    with gr.Blocks(title="Talk to Adrishuo - a voice-enabled assistant working as a healthcare assistant") as demo:
        gr.Markdown("""
        # Talk to Adrishuo - a voice-enabled assistant working as a healthcare assistant

        Instructions for use:
        - record your question/comment using the first audio widget ("Your voice input")
        - wait for the chatbot to response ("Chatbot")
        - click summarize button to make a summary
        """)
        with gr.Row():
            # user's input
            input_audio_ui = gr.Audio(sources=["microphone"], scale=5, label="Your voice input")
            # submit button
            submit_audio_btn = gr.Button("Submit", variant="primary", scale=1)

        # chatbot
        chatbot_ui = gr.Chatbot(value=[[None, initial_message]], label="Chatbot")

        # summarize
        summarize_button = gr.Button("Summarize", variant="primary")
        summary_ui = gr.Textbox(label="Summary", interactive=False)

        # events
        submit_audio_btn.click(transcribe, inputs=[input_audio_ui, chatbot_ui], outputs=chatbot_ui)\
            .then(chat, chatbot_ui, chatbot_ui)\
            .then(lambda: None, inputs=[], outputs=[input_audio_ui])

        summarize_button.click(summarize, inputs=chatbot_ui, outputs=summary_ui)

    return demo


def run(asr_model_dir: Path, chat_model_dir: Path, public_interface: bool = False) -> None:
    """
    Run the assistant application

    Params
        asr_model_dir: dir with the automatic speech recognition model
        chat_model_dir: dir with the chat model
        public_interface: whether UI should be available publicly
    """
    # load whisper model
    load_asr_model(asr_model_dir)
    # load chat model
    load_chat_model(chat_model_dir)

    # get initial greeting
    initial_message = generate_initial_greeting()

    # create user interface
    demo = create_UI(initial_message)
    # launch demo
    demo.launch(share=public_interface)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asr_model_dir', type=str, default="model/distil-large-v2-FP16", help="Path to the automatic speech recognition model directory")
    parser.add_argument('--chat_model_dir', type=str, default="model/llama2-7B-INT8", help="Path to the chat model directory")
    parser.add_argument('--public_interface', default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run(Path(args.asr_model_dir), Path(args.chat_model_dir), args.public_interface)
