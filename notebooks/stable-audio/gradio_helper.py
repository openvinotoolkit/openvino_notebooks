from typing import Callable
import gradio as gr
import numpy as np


MAX_SEED = np.iinfo(np.int32).max


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Textbox(label="Text Prompt"),
            gr.Slider(1, 47, label="Total seconds", step=1, value=10),
            gr.Slider(10, 100, label="Number of steps", step=1, value=100),
            gr.Slider(0, MAX_SEED, label="Seed", step=1),
        ],
        outputs=["audio"],
        examples=[
            ["128 BPM tech house drum loop"],
            ["Blackbird song, summer, dusk in the forest"],
            ["Rock beat played in a treated studio, session drumming on an acoustic kit"],
            ["Calmful melody and nature sounds for restful sleep"],
        ],
        allow_flagging="never",
    )
    return demo
