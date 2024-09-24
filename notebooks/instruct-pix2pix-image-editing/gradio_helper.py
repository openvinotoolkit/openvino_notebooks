from pathlib import Path
from typing import Callable
import gradio as gr
import requests

example_image_url = "https://user-images.githubusercontent.com/29454499/223343459-4ac944f0-502e-4acf-9813-8e9f0abc8a16.jpg"
example_image_path = Path("data/example.jpg")

if not example_image_path.exists():
    example_image_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(example_image_url)
    with example_image_path.open("wb") as f:
        f.write(r.content)


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Image(label="Image", type="pil"),
            gr.Textbox(label="Text"),
            gr.Slider(0, 1024, label="Seed", value=42),
            gr.Slider(
                1,
                100,
                label="Steps",
                value=10,
                info="Consider increasing the value to get more precise results. A suggested value is 100, but it will take more time to process.",
            ),
        ],
        outputs=gr.Image(label="Result"),
        examples=[[example_image_path, "Make it in galaxy"]],
    )
    return demo
