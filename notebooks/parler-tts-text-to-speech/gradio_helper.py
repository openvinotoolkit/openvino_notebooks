from typing import Callable
import gradio as gr
import numpy as np


title = "Text-to-speech (TTS) with Parler-TTS and OpenVINO"

examples = [
    [
        "Hey, how are you doing today?",
        "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast.",
    ],
    [
        "'This is the best time of my life, Bartley,' she said happily.",
        "A female speaker with a slightly low-pitched, quite monotone voice delivers her words at a slightly faster-than-average pace in a confined space with very clear audio.",
    ],
    [
        "Montrose also, after having experienced still more variety of good and bad fortune, threw down his arms, and retired out of the kingdom.	",
        "A male speaker with a slightly high-pitched voice delivering his words at a slightly slow pace in a small, confined space with a touch of background noise and a quite monotone tone.",
    ],
    [
        "montrose also after having experienced still more variety of good and bad fortune threw down his arms and retired out of the kingdom",
        "A male speaker with a low-pitched voice delivering his words at a fast pace in a small, confined space with a lot of background noise and an animated tone.",
    ],
]


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Text(label="Prompt"),
            gr.Text(label="Description"),
            gr.Slider(
                label="Seed",
                value=42,
                step=1,
                minimum=0,
                maximum=np.iinfo(np.int32).max,
            ),
        ],
        outputs=gr.Audio(label="Output Audio", type="numpy"),
        title=title,
        examples=examples,
    )
    return demo
