from typing import Callable
import gradio as gr
from functools import partial


def make_demo(fn: Callable, quantized: bool):
    with gr.Blocks() as demo:
        with gr.Row(equal_height=False):
            image = gr.Image(type="filepath")
            with gr.Column():
                output_image = gr.Image(show_label=True, type="filepath", interactive=False, label="FP16 model output")
                button = gr.Button(value="Run FP16 model")
            with gr.Column(visible=quantized):
                output_image_int8 = gr.Image(show_label=True, type="filepath", interactive=False, label="INT8 model output")
                button_i8 = gr.Button(value="Run INT8 model")
        button.click(fn=partial(fn, use_int8=False), inputs=[image], outputs=[output_image])
        button_i8.click(fn=partial(fn, use_int8=True), inputs=[image], outputs=[output_image_int8])
        examples = gr.Examples(
            [
                "DDColor/assets/test_images/New York Riverfront December 15, 1931.jpg",
                "DDColor/assets/test_images/Audrey Hepburn.jpg",
                "DDColor/assets/test_images/Acrobats Balance On Top Of The Empire State Building, 1934.jpg",
            ],
            inputs=[image],
        )
    return demo
