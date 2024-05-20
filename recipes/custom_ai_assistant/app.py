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
TARGET_AUDIO_SAMPLE_RATE = 16000
SYSTEM_CONFIGURATION = (
    "You are Adrishuo - a helpful, respectful, and honest virtual doctor assistant. "
    "Your role is talking to a patient who just came in."
    "Your primary role is to assist in the collection of Symptom information from patients. "
    "Focus solely on gathering symptom details without offering treatment or medical advice."
    "You must only ask follow-up questions based on the patient's initial descriptions to clarify and gather more details about their symtpoms. "
    "You must not attempt to diagnose, treat, or offer health advice. "
    "Ask one and only the symptom related followup questions and keep it short. "
    "You must strictly not suggest or recommend any treatments, including over-the-counter medication. "
    "You must strictly avoid making any assumptions or conclusions about the causes or nature of the patient's symptoms. "
    "You must strictly avoid providing suggestions to manage their symptoms. "
    "Your interactions should be focused solely on understanding and recording the patient's stated symptoms."
    "Do not collect or use any personal information like age, name, contact, gender, etc."
    "Remember, your role is to aid in symptom information collection in a supportive, unbiased, and factually accurate manner. "
    "Your responses should consistently encourage the patient to discuss their symptoms in greater detail while remaining neutral and non-diagnostic."
)
GREET_THE_CUSTOMER = "Please introduce yourself and greet the patient"
SUMMARIZE_THE_CUSTOMER = (
    "You are now required to summarize the patient's exact provided symptoms for the doctor's review. "
    "Strictly do not mention any personal data like age, name, gender, contact, non-health information etc. when summarizing."
    "Warn the patients for immediate medical seeking in case they exhibit symptoms indicative of critical conditions such as heart attacks, strokes, severe allergic reactions, breathing difficulties, high fever with severe symptoms, significant burns, or severe injuries."
    "Summarize the health-related concerns mentioned by the patient in this conversation, focusing only on the information explicitly provided, without adding any assumptions or unrelated symptoms."
)

# Initialize Model variables
chat_model: Optional[OVModelForCausalLM] = None
chat_tokenizer: Optional[PreTrainedTokenizer] = None
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
    global chat_model, chat_tokenizer

    # load llama model and its tokenizer
    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    chat_model = OVModelForCausalLM.from_pretrained(model_dir, device="AUTO", config=AutoConfig.from_pretrained(model_dir), ov_config=ov_config)
    chat_tokenizer = AutoTokenizer.from_pretrained(model_dir)


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
    tokens = outputs[0, input_length:]
    end_time = time.time()  # End time
    print("Chat model response time: {:.2f} seconds".format(end_time - start_time))
    # decode tokens into text
    return chat_tokenizer.decode(tokens, skip_special_tokens=True)


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
    return chat_tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


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
    chat_streamer = TextIteratorStreamer(chat_tokenizer, skip_prompt=True, skip_special_tokens=True)

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
    audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_AUDIO_SAMPLE_RATE).astype(np.int16)

    # get input features from the audio
    input_features = asr_processor(audio, sampling_rate=TARGET_AUDIO_SAMPLE_RATE, return_tensors="pt").input_features

    # use streamer to show transcription word by word
    text_streamer = TextIteratorStreamer(asr_processor, skip_prompt=True, skip_special_tokens=True)

    # transcribe in the background to deliver response token by token
    thread = Thread(target=asr_model.generate, kwargs={"input_features": input_features, "streamer": text_streamer})
    thread.start()

    conversation.append(["", None])
    # get token by token and merge to the final response
    for partial_text in text_streamer:
        conversation[-1][0] += partial_text
        # "return" partial response
        yield conversation

    end_time = time.time()  # End time for ASR process
    print(f"ASR model response time: {end_time - start_time:.2f} seconds")  # Print the ASR processing time

    # wait for the thread
    thread.join()

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
        yield partial_summary[-1][1]


def create_UI(initial_message: str) -> gr.Blocks:
    """
    Create web user interface

    Params:
        initial_message: message to start with
    Returns:
        Demo UI
    """
    with gr.Blocks(title="Talk to Adrishuo - a custom AI assistant working as a healthcare assistant") as demo:
        gr.Markdown("""
        # Talk to Adrishuo - a custom AI assistant working today as a healthcare assistant

        Instructions for use:
        - record your question/comment using the first audio widget ("Your voice input")
        - wait for the chatbot to response ("Chatbot")
        - click summarize button to make a summary
        """)
        with gr.Row():
            # user's input
            input_audio_ui = gr.Audio(sources=["microphone"], scale=5, label="Your voice input")
            # submit button
            submit_audio_btn = gr.Button("Submit", variant="primary", scale=1, interactive=False)

        # chatbot
        chatbot_ui = gr.Chatbot(value=[[None, initial_message]], label="Chatbot")

        # summarize
        summarize_button = gr.Button("Summarize", variant="primary", interactive=False)
        summary_ui = gr.Textbox(label="Summary", interactive=False)

        # events
        # block submit button when no audio input
        input_audio_ui.change(lambda x: gr.Button(interactive=False) if x is None else gr.Button(interactive=True), inputs=input_audio_ui, outputs=submit_audio_btn)

        # block buttons, do the transcription and conversation, clear audio, unblock buttons
        submit_audio_btn.click(lambda: gr.Button(interactive=False), outputs=submit_audio_btn) \
            .then(lambda: gr.Button(interactive=False), outputs=summarize_button)\
            .then(transcribe, inputs=[input_audio_ui, chatbot_ui], outputs=chatbot_ui)\
            .then(chat, chatbot_ui, chatbot_ui)\
            .then(lambda: None, inputs=[], outputs=[input_audio_ui])\
            .then(lambda: gr.Button(interactive=True), outputs=summarize_button)

        # block button, do the summarization, unblock button
        summarize_button.click(lambda: gr.Button(interactive=False), outputs=summarize_button) \
            .then(summarize, inputs=chatbot_ui, outputs=summary_ui) \
            .then(lambda: gr.Button(interactive=True), outputs=summarize_button)

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
    parser.add_argument('--asr_model_dir', type=str, default="model/distil-whisper-large-v2-FP16", help="Path to the automatic speech recognition model directory")
    parser.add_argument('--chat_model_dir', type=str, default="model/llama3-8B-INT8", help="Path to the chat model directory")
    parser.add_argument('--public_interface', default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run(Path(args.asr_model_dir), Path(args.chat_model_dir), args.public_interface)
