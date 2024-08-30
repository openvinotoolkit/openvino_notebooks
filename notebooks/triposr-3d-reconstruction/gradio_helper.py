import os
from typing import Callable
import gradio as gr

examples = [os.path.join("TripoSR/examples", img_name) for img_name in sorted(os.listdir("TripoSR/examples"))]


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def make_demo(preprocess_fn: Callable, generate_fn: Callable):
    with gr.Blocks() as demo:
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(
                        label="Input Image",
                        image_mode="RGBA",
                        sources="upload",
                        type="pil",
                        elem_id="content_image",
                    )
                    processed_image = gr.Image(label="Processed Image", interactive=False)
                with gr.Row():
                    with gr.Group():
                        do_remove_background = gr.Checkbox(label="Remove Background", value=True)
                        foreground_ratio = gr.Slider(
                            label="Foreground Ratio",
                            minimum=0.5,
                            maximum=1.0,
                            value=0.85,
                            step=0.05,
                        )
                with gr.Row():
                    submit = gr.Button("Generate", elem_id="generate", variant="primary")
            with gr.Column():
                with gr.Tab("Model"):
                    output_model = gr.Model3D(
                        label="Output Model",
                        interactive=False,
                    )
        with gr.Row(variant="panel"):
            gr.Examples(
                examples=examples,
                inputs=[input_image],
                outputs=[processed_image, output_model],
                label="Examples",
                examples_per_page=20,
            )
        submit.click(fn=check_input_image, inputs=[input_image]).success(
            fn=preprocess_fn,
            inputs=[input_image, do_remove_background, foreground_ratio],
            outputs=[processed_image],
        ).success(
            fn=generate_fn,
            inputs=[processed_image],
            outputs=[output_model],
        )
    return demo
