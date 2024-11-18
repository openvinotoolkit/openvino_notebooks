from pathlib import Path
import requests
from PIL import Image
import gradio as gr

MARKDOWN = """
# OpenVINO OmniParser for Pure Vision Based General GUI Agent 🔥
OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

example_images = [
    ("https://github.com/microsoft/OmniParser/blob/master/imgs/windows_home.png?raw=true", "examples/windows_home.png"),
    ("https://github.com/microsoft/OmniParser/blob/master/imgs/logo.png?raw=true", "examples/logo.png"),
    ("https://github.com/microsoft/OmniParser/blob/master/imgs/windows_multitab.png?raw=true", "examples/multitab.png"),
]


def make_demo(process_fn):
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True, parents=True)
    for url, filename in example_images:
        if not Path(filename).exists():
            image = Image.open(requests.get(url, stream=True).raw)
            image.save(filename)

    with gr.Blocks() as demo:
        gr.Markdown(MARKDOWN)
        with gr.Row():
            with gr.Column():
                image_input_component = gr.Image(type="filepath", label="Upload image")
                # set the threshold for removing the bounding boxes with low confidence, default is 0.05
                box_threshold_component = gr.Slider(label="Box Threshold", minimum=0.01, maximum=1.0, step=0.01, value=0.05)
                # set the threshold for removing the bounding boxes with large overlap, default is 0.1
                iou_threshold_component = gr.Slider(label="IOU Threshold", minimum=0.01, maximum=1.0, step=0.01, value=0.1)
                imgsz_component = gr.Slider(label="Icon Detect Image Size", minimum=640, maximum=1920, step=32, value=640)
                submit_button_component = gr.Button(value="Submit", variant="primary")
            with gr.Column():
                image_output_component = gr.Image(type="pil", label="Image Output")
                text_output_component = gr.Textbox(label="Parsed screen elements", placeholder="Text Output")
        gr.Examples(
            examples=list(Path("examples").glob("*.png")),
            inputs=[image_input_component],
            label="Try examples",
        )
        submit_button_component.click(
            fn=process_fn,
            inputs=[image_input_component, box_threshold_component, iou_threshold_component, imgsz_component],
            outputs=[image_output_component, text_output_component],
        )
    return demo
