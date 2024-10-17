import os
from typing import Callable
import gradio as gr


ref_dir = "data/reference"
image_dir = "data/image"
ref_list = [os.path.join(ref_dir, file) for file in os.listdir(ref_dir) if file.endswith(".jpg")]
ref_list.sort()
image_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(".png")]
image_list.sort()


def make_demo(fn: Callable):
    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    image = gr.Image(
                        source="upload",
                        tool="sketch",
                        elem_id="image_upload",
                        type="pil",
                        label="Source Image",
                    )
                    reference = gr.Image(
                        source="upload",
                        elem_id="image_upload",
                        type="pil",
                        label="Reference Image",
                    )

                with gr.Column():
                    image_out = gr.Image(label="Output", elem_id="output-img")
                    steps = gr.Slider(
                        label="Steps",
                        value=15,
                        minimum=2,
                        maximum=75,
                        step=1,
                        interactive=True,
                    )
                    seed = gr.Slider(0, 10000, label="Seed (0 = random)", value=0, step=1)

                    with gr.Row(elem_id="prompt-container"):
                        btn = gr.Button("Paint!")

            with gr.Row():
                with gr.Column():
                    gr.Examples(
                        image_list,
                        inputs=[image],
                        label="Examples - Source Image",
                        examples_per_page=12,
                    )
                with gr.Column():
                    gr.Examples(
                        ref_list,
                        inputs=[reference],
                        label="Examples - Reference Image",
                        examples_per_page=12,
                    )

            btn.click(
                fn=fn,
                inputs=[image, reference, seed, steps],
                outputs=[image_out],
            )
    return demo
