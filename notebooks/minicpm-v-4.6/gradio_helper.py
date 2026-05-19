from pathlib import Path

import gradio as gr
from PIL import Image
import numpy as np
import requests
from threading import Event, Thread
import inspect

example_image_urls = [
    (
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1d6a0188-5613-418d-a1fd-4560aae1d907",
        "bee.jpg",
    ),
    (
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/6cc7feeb-0721-4b5d-8791-2576ed9d2863",
        "baklava.png",
    ),
]
for url, file_name in example_image_urls:
    if not Path(file_name).exists():
        Image.open(requests.get(url, stream=True, timeout=30).raw).save(file_name)


def make_demo(ov_model, processor):
    from transformers import TextIteratorStreamer

    has_additonal_buttons = "undo_button" in inspect.signature(gr.ChatInterface.__init__).parameters

    tokenizer = processor.tokenizer

    def bot_streaming(message, history):
        files = message["files"] if isinstance(message, dict) else message.files
        message_text = message["text"] if isinstance(message, dict) else message.text

        image = None
        if files:
            if isinstance(files[-1], dict):
                image = files[-1]["path"]
            else:
                if isinstance(files[-1], (str, Path)):
                    image = files[-1]
                else:
                    image = files[-1] if isinstance(files[-1], (list, tuple)) else files[-1].path
            image = Image.open(image).convert("RGB")

        inputs = ov_model.preprocess_inputs(
            text=message_text,
            image=image,
            processor=processor,
            downsample_mode="16x",
            max_slice_nums=36,
        )

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        stream_complete = Event()

        def generate_and_signal_complete():
            generate_kwargs = dict(
                **inputs,
                downsample_mode="16x",
                max_new_tokens=512,
                do_sample=False,
                streamer=streamer,
            )
            ov_model.generate(**generate_kwargs)
            stream_complete.set()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            yield buffer

    additional_buttons = {}
    if has_additonal_buttons:
        additional_buttons = {"undo_button": None, "retry_button": None}
    demo = gr.ChatInterface(
        fn=bot_streaming,
        title="MiniCPM-V 4.6 OpenVINO Chatbot",
        examples=[
            {"text": "What is on the flower?", "files": ["./bee.jpg"]},
            {"text": "How to make this pastry?", "files": ["./baklava.png"]},
        ],
        stop_btn=None,
        multimodal=True,
        **additional_buttons,
    )
    return demo
