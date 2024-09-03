from typing import Callable
import gradio as gr


main_model_id = "meta-llama/Llama-2-7b-chat-hf"
draft_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def make_demo(fn: Callable):
    with gr.Blocks() as demo:
        gr.Markdown(
            f"""
            # Speculative Sampling Demo
            ## The output will show a comparison of Autoregressive Sampling vs Speculative Sampling
            - Main Model: {main_model_id}
            - Draft Model: {draft_model_id}
            - K = 5
            """
        )
        with gr.Row():
            input = gr.Textbox(
                value="Alan Turing was a",
                placeholder="THIS CANNOT BE EMPTY",
                label="Input Prompt",
            )
            output = gr.Textbox(label="Output")
        btn = gr.Button("Run")
        btn.click(fn=fn, inputs=input, outputs=output)
    return demo
