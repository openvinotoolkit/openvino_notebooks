from typing import Callable
import gradio as gr
import os
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


def make_demo(fn: Callable):
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# Depth Anything with OpenVINO")
        gr.Markdown("### Depth Prediction demo")
        gr.Markdown("You can slide the output to compare the depth prediction with input image")

        with gr.Row():
            input_image = gr.Image(label="Input Image", type="numpy", elem_id="img-display-input")
            depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id="img-display-output", position=0)
        raw_file = gr.File(label="16-bit raw depth (can be considered as disparity)")
        submit = gr.Button("Submit")

        submit.click(fn=fn, inputs=[input_image], outputs=[depth_image_slider, raw_file])

        example_files = os.listdir("assets/examples")
        example_files.sort()
        example_files = [os.path.join("assets/examples", filename) for filename in example_files]
        examples = gr.Examples(
            examples=example_files,
            inputs=[input_image],
            outputs=[depth_image_slider, raw_file],
            fn=fn,
            cache_examples=False,
        )
    return demo
