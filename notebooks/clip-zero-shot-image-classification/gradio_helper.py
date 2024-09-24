from pathlib import Path
import gradio as gr
import requests


sample_path = Path("data/coco.jpg")

if not sample_path.exists():
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg")
    with sample_path.open("wb") as f:
        f.write(r.content)


def make_demo(classify):
    demo = gr.Interface(
        classify,
        [
            gr.Image(label="Image", type="pil"),
            gr.Textbox(label="Labels", info="Comma-separated list of class labels"),
        ],
        gr.Label(label="Result"),
        examples=[[sample_path, "cat,dog,bird"]],
    )

    return demo
