from typing import Callable
import gradio as gr
from functools import partial

default_text = (
    "Most of the course is about semantic or  content of language but there are also interesting"
    " topics to be learned from the servicefeatures except statistics in characters in documents.At"
    " this point, He introduces herself as his native English speaker and goes on to say that if"
    " you contine to work on social scnce"
)


def make_demo(fn: Callable, quantized: bool):
    model_type = "optimized" if quantized else "original"
    with gr.Blocks() as demo:
        gr.Markdown("# Interactive demo")
        with gr.Row():
            gr.Markdown(f"## Run {model_type} grammar correction pipeline")
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Text")
            with gr.Column():
                output_text = gr.Textbox(label="Correction", interactive=False, show_copy_button=True)
                correction_time = gr.Textbox(label="Time (seconds)", interactive=False)
        with gr.Row():
            gr.Examples(examples=[default_text], inputs=[input_text])
        with gr.Row():
            button = gr.Button(f"Run {model_type}")
            button.click(
                fn=partial(fn, quantized=quantized),
                inputs=[input_text, gr.Number(quantized, visible=False)],
                outputs=[output_text, correction_time],
            )
    return demo
