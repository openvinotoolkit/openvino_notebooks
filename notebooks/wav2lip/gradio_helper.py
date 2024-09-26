from typing import Callable
import gradio as gr
import numpy as np


examples = [
    [
        "data_video_sun_5s.mp4",
        "data_audio_sun_5s.wav",
    ],
]


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Video(label="Face video"),
            gr.Audio(label="Audio", type="filepath"),
        ],
        outputs="video",
        examples=examples,
        allow_flagging="never",
    )
    return demo
