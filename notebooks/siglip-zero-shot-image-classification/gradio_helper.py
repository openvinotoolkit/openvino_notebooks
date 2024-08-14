from pathlib import Path
import gradio as gr
import requests

image_path = Path("test_image.jpg")

if not image_path.exists():
    r = requests.get(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
    )
    with image_path.open("wb") as f:
        f.write(r.content)


def make_demo(fn):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Image(label="Image", type="pil"),
            gr.Textbox(label="Labels", info="Comma-separated list of class labels"),
        ],
        outputs=gr.Label(label="Result"),
        examples=[[image_path, "cat,dog,bird"]],
    )

    return demo
