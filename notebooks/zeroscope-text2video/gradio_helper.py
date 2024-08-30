from typing import Callable
import gradio as gr


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Textbox(label="Prompt"),
            gr.Slider(0, 1000000, value=42, label="Seed", step=1),
            gr.Slider(10, 50, value=25, label="Number of inference steps", step=1),
        ],
        outputs=gr.Image(label="Result"),
        examples=[
            ["An astronaut riding a horse.", 0, 25],
            ["A panda eating bamboo on a rock.", 0, 25],
            ["Spiderman is surfing.", 0, 25],
        ],
        allow_flagging="never",
    )
    return demo
