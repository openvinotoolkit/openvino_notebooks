from pathlib import Path
from typing import List, TypedDict
import gradio as gr

sample_path = Path("data")


class Option(TypedDict):
    choices: List[str]
    value: str


def make_demo(run, model_option: Option, device_option: Option):
    desc_text = """
    Search the content's of a video with a text description.
    __Note__: Long videos (over a few minutes) may cause UI performance issues.
        """
    text_app = gr.Interface(
        description=desc_text,
        fn=run,
        inputs=[
            gr.Video(label="Video"),
            gr.Textbox(label="Text Search Query"),
            gr.Image(label="Image Search Query", visible=False),
            gr.Dropdown(
                label="Model",
                choices=model_option["choices"],
                value=model_option["value"],
            ),
            gr.Dropdown(label="Device", choices=device_option["choices"], value=device_option["value"]),
            gr.Slider(label="Threshold", maximum=1.0, value=0.2),
            gr.Slider(label="Frame-rate Stride", value=4, step=1),
            gr.Slider(label="Batch Size", value=4, step=1),
        ],
        outputs=[
            gr.Plot(label="Similarity Plot"),
            gr.Gallery(label="Matched Frames", columns=2, object_fit="contain", height="auto"),
        ],
        examples=[[sample_path / "car-detection.mp4", "white car"]],
        allow_flagging="never",
    )

    desc_image = """
    Search the content's of a video with an image query.
    __Note__: Long videos (over a few minutes) may cause UI performance issues.
        """
    image_app = gr.Interface(
        description=desc_image,
        fn=run,
        inputs=[
            gr.Video(label="Video"),
            gr.Textbox(label="Text Search Query", visible=False),
            gr.Image(label="Image Search Query", type="pil"),
            gr.Dropdown(
                label="Model",
                choices=model_option["choices"],
                value=model_option["value"],
            ),
            gr.Dropdown(label="Device", choices=device_option["choices"], value=device_option["value"]),
            gr.Slider(label="Threshold", maximum=1.0, value=0.2),
            gr.Slider(label="Frame-rate Stride", value=4, step=1),
            gr.Slider(label="Batch Size", value=4, step=1),
        ],
        outputs=[
            gr.Plot(label="Similarity Plot"),
            gr.Gallery(label="Matched Frames", columns=2, object_fit="contain", height="auto"),
        ],
        allow_flagging="never",
        examples=[[sample_path / "coco.mp4", None, sample_path / "dog.png"]],
    )
    demo = gr.TabbedInterface(
        interface_list=[text_app, image_app],
        tab_names=["Text Query Search", "Image Query Search"],
        title="CLIP Video Content Search",
    )

    return demo
