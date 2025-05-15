from typing import Callable
import gradio as gr
import numpy as np

MAX_SEED = np.iinfo(np.int32).max


def make_demo_lcm(fn: Callable):
    examples = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour,"
        "style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
    ]
    MAX_IMAGE_SIZE = 768
    gr.close_all()
    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
            with gr.Row():
                with gr.Column():
                    result = gr.Image(
                        label="Result Image",
                        type="pil",
                    )
                    run_button = gr.Button("Run")

        with gr.Accordion("Advanced options", open=False):
            seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True)
            randomize_seed = gr.Checkbox(label="Randomize seed across runs", value=True)
            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale for base",
                    minimum=2,
                    maximum=14,
                    step=0.1,
                    value=8.0,
                )
                num_inference_steps = gr.Slider(
                    label="Number of inference steps for base",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=4,
                )

        gr.Examples(
            examples=examples,
            inputs=prompt,
            outputs=result,
            cache_examples=False,
        )

        gr.on(
            triggers=[
                prompt.submit,
                run_button.click,
            ],
            fn=fn,
            inputs=[
                prompt,
                seed,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                randomize_seed,
            ],
            outputs=[result, seed],
        )
    return demo
