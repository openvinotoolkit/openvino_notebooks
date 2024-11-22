from pathlib import Path
from typing import Callable
import gradio as gr
from gradio_imageslider import ImageSlider

css = """
#img-display-container {
    max-height: 100vh;
    }
#img-display-input {
    max-height: 80vh;
    }
#img-display-output {
    max-height: 80vh;
    }
"""


def make_demo(fn: Callable, examples_dir: str):
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# Depth Anything with OpenVINO")
        gr.Markdown("### Depth Prediction demo")
        gr.Markdown("You can slide the output to compare the depth prediction with input image")

        with gr.Row():
            input_image = gr.Image(label="Input Image", type="numpy", elem_id="img-display-input")
            depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id="img-display-output", position=0)
        depth_image_file = gr.File(label="Depth Image")
        submit = gr.Button("Submit")

        submit.click(fn=fn, inputs=[input_image], outputs=[depth_image_slider, depth_image_file])

        if not Path(examples_dir).exists():
            gr.Error(f"Examples directory {examples_dir} does not exist.")
            raise FileNotFoundError(f"Examples directory {examples_dir} does not exist.")
        example_files = sorted([str(image_path) for image_path in Path(examples_dir).iterdir()])
        examples = gr.Examples(
            examples=example_files,
            inputs=[input_image],
            outputs=[depth_image_slider, depth_image_file],
            fn=fn,
            cache_examples=False,
        )
    return demo
