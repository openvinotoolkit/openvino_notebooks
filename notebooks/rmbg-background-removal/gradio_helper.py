from typing import Callable
import gradio as gr


def make_demo(fn: Callable):
    with gr.Blocks() as demo:
        gr.Markdown("# RMBG background removal with OpenVINO")

        with gr.Row():
            input_image = gr.Image(label="Input Image", type="numpy")
            background_image = gr.Image(label="Background removal Image")
        submit = gr.Button("Submit")

        submit.click(fn=fn, inputs=[input_image], outputs=[background_image])

        gr.Examples(
            examples=["./example_input.jpg"],
            inputs=[input_image],
            outputs=[background_image],
            fn=fn,
            cache_examples=False,
        )
    return demo
