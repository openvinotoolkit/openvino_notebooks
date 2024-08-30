from typing import Callable
import gradio as gr


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Textbox(label="Text Prompt"),
            gr.Textbox(label="Negative Prompt", placeholder="Example: Low quality"),
            gr.Slider(
                minimum=1.0,
                maximum=15.0,
                step=0.25,
                value=7,
                label="Audio Length (s)",
            ),
            gr.Slider(label="Inference Steps", step=5, value=150, minimum=50, maximum=250),
        ],
        outputs=["audio"],
        examples=[
            ["birds singing in the forest", "Low quality", 7, 150],
            ["The sound of a hammer hitting a wooden surface", "", 4, 200],
        ],
    )
    return demo
