from typing import Callable
import gradio as gr
import numpy as np

MAX_SEED = np.iinfo(np.int32).max


def make_demo_lcm_lora(fn: Callable, quantized: bool):
    gr.close_all()
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                inp_img = gr.Image(label="Input image")
            with gr.Column(visible=True):
                out_normal = gr.Image(label="Normal Map", type="pil", interactive=False)
                btn = gr.Button()
                inp_prompt = gr.Textbox(label="Prompt")
                inp_neg_prompt = gr.Textbox(
                    "",
                    label="Negative prompt",
                )
                with gr.Accordion("Advanced options", open=False):
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=2,
                        step=0.1,
                        value=0.5,
                    )
                    inp_seed = gr.Slider(label="Seed", value=42, maximum=MAX_SEED)
                    inp_steps = gr.Slider(label="Steps", value=4, minimum=1, maximum=50, step=1)
            with gr.Column(visible=True):
                out_result = gr.Image(label="Result (Original)")
            with gr.Column(visible=quantized):
                int_result = gr.Image(label="Result (Quantized)")
        gr.Examples([["example.png", "a head full of roses"]], [inp_img, inp_prompt])

        output_images = [out_normal, out_result]
        if quantized:
            output_images.append(int_result)
        btn.click(
            fn=fn,
            inputs=[inp_img, inp_prompt, inp_neg_prompt, inp_seed, inp_steps, guidance_scale],
            outputs=output_images,
        )
    return demo
