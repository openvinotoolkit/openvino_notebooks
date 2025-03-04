from copy import deepcopy
from typing import Dict, List
from PIL import Image
import librosa
from transformers import TextIteratorStreamer
from threading import Thread
import gradio as gr

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
AUDIO_EXTENSIONS = (".mp3", ".wav", "flac", ".m4a", ".wma")

IMAGE_SPECIAL = "<|endoftext10|>"
AUDIO_SPECIAL = "<|endoftext11|>"

DEFAULT_SAMPLING_PARAMS = {
    "top_p": 0.0,
    "top_k": 1,
    "temperature": 0.0,
    "do_sample": True,
    "num_beams": 1,
    "repetition_penalty": 1.2,
}
MAX_NEW_TOKENS = 512


def history2messages(history: List[Dict]) -> List[Dict]:
    """
    Transform gradio history to chat messages.
    """
    messages = []
    cur_message = dict()
    images = []
    audios = []
    cur_special_tags = ""
    for item in history:
        if item["role"] == "assistant":
            if len(cur_message) > 0:
                cur_message["content"] = cur_special_tags + cur_message["content"]
                messages.append(deepcopy(cur_message))
                cur_message = dict()
                cur_special_tags = ""
            messages.append({"role": "assistant", "content": item["content"]})
            continue

        if "role" not in cur_message:
            cur_message["role"] = "user"
        if "content" not in cur_message:
            cur_message["content"] = ""

        if "metadata" not in item:
            item["metadata"] = {"title": None}
        if item["metadata"].get("title") is None:
            cur_message["content"] = item["content"]
        elif item["metadata"]["title"] == "image":
            cur_special_tags += IMAGE_SPECIAL
            images.append(Image.open(item["content"][0]))
        elif item["metadata"]["title"] == "audio":
            cur_special_tags += AUDIO_SPECIAL
            audios.append(librosa.load(item["content"][0]))
    if len(cur_message) > 0:
        cur_message["content"] = cur_special_tags + cur_message["content"]
        messages.append(cur_message)
    return messages, images, audios


def check_messages(history, message, audio):
    has_text = message["text"] and message["text"].strip()
    has_files = len(message["files"]) > 0
    has_audio = audio is not None

    if not (has_text or has_files or has_audio):
        raise gr.Error("Message is empty")

    audios = []
    images = []

    for file_msg in message["files"]:
        if file_msg.endswith(AUDIO_EXTENSIONS):
            duration = librosa.get_duration(filename=file_msg)
            if duration > 60:
                raise gr.Error("Audio file too long. For efficiency we recommend to use audio < 60s")
            if duration == 0:
                raise gr.Error("Audio file too short")
            audios.append(file_msg)
        elif file_msg.endswith(IMAGE_EXTENSIONS):
            images.append(file_msg)
        else:
            filename = file_msg.split("/")[-1]
            raise gr.Error(f"Unsupported file type: {filename}. It should be an image or audio file.")

    if len(audios) > 1:
        raise gr.Error("Please upload only one audio file.")

    if len(images) > 1:
        raise gr.Error("Please upload only one image file.")

    if audio is not None:
        if len(audios) > 0:
            raise gr.Error("Please upload only one audio file or record audio.")
        audios.append(audio)

    # Append the message to the history
    for image in images:
        history.append({"role": "user", "content": (image,), "metadata": {"title": "image"}})

    for audio in audios:
        history.append({"role": "user", "content": (audio,), "metadata": {"title": "audio"}})

    if message["text"]:
        history.append({"role": "user", "content": message["text"], "metadata": {}})

    return history, gr.MultimodalTextbox(value=None, interactive=False), None


def make_demo(ov_model, processor):
    def bot(
        history: list,
        top_p: float,
        top_k: int,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int = MAX_NEW_TOKENS,
        regenerate: bool = False,
    ):

        if history and regenerate:
            history = history[:-1]

        if not history:
            return history

        msgs, images, audios = history2messages(history)
        audios = audios if len(audios) > 0 else None
        images = images if len(images) > 0 else None
        prompt = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, audios=audios, images=images)
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_params = {
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "streamer": streamer,
            **inputs,
        }

        history.append({"role": "assistant", "content": ""})

        thread = Thread(target=ov_model.generate, kwargs=generation_params)
        thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            history[-1]["content"] = buffer
            yield history

    def change_state(state):
        return gr.update(visible=not state), not state

    def reset_user_input():
        return gr.update(value="")

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🪐 Chat with OpenVINO Phi-4-multimodal")
        chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages", height="48vh")

        sampling_params_group_hidden_state = gr.State(False)

        with gr.Row(equal_height=True):
            chat_input = gr.MultimodalTextbox(
                file_count="multiple",
                placeholder="Enter your prompt or upload image/audio here, then press ENTER...",
                show_label=False,
                scale=8,
                file_types=["image", "audio"],
                interactive=True,
                # stop_btn=True,
            )
        with gr.Row(equal_height=True):
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", scale=1, max_length=30)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=150):
                with gr.Row(equal_height=True):
                    regenerate_btn = gr.Button("Regenerate", variant="primary")
                    clear_btn = gr.ClearButton([chat_input, audio_input, chatbot])

        with gr.Row():
            sampling_params_toggle_btn = gr.Button("Sampling Parameters")

        with gr.Group(visible=False) as sampling_params_group:
            with gr.Row():
                temperature = gr.Slider(minimum=0, maximum=1, value=DEFAULT_SAMPLING_PARAMS["temperature"], label="Temperature")
                repetition_penalty = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=DEFAULT_SAMPLING_PARAMS["repetition_penalty"],
                    label="Repetition Penalty",
                )

            with gr.Row():
                top_p = gr.Slider(minimum=0, maximum=1, value=DEFAULT_SAMPLING_PARAMS["top_p"], label="Top-p")
                top_k = gr.Slider(minimum=0, maximum=1000, value=DEFAULT_SAMPLING_PARAMS["top_k"], label="Top-k")

            with gr.Row():
                max_new_tokens = gr.Slider(
                    minimum=1,
                    maximum=MAX_NEW_TOKENS,
                    value=MAX_NEW_TOKENS,
                    label="Max New Tokens",
                    interactive=True,
                )

        sampling_params_toggle_btn.click(
            change_state,
            sampling_params_group_hidden_state,
            [sampling_params_group, sampling_params_group_hidden_state],
        )
        chat_msg = chat_input.submit(
            check_messages,
            [chatbot, chat_input, audio_input],
            [chatbot, chat_input, audio_input],
        )

        bot_msg = chat_msg.then(
            bot,
            inputs=[chatbot, top_p, top_k, temperature, repetition_penalty, max_new_tokens],
            outputs=chatbot,
        )

        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        regenerate_btn.click(
            bot,
            inputs=[chatbot, top_p, top_k, temperature, repetition_penalty, max_new_tokens, gr.State(True)],
            outputs=chatbot,
        )
    return demo
