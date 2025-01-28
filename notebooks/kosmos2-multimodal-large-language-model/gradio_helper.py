from PIL import Image
from typing import Callable
import gradio as gr
from pathlib import Path
import requests

images = {
    "snowman.png": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png",
    "two_dogs.jpg": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/two_dogs.jpg",
    "six_planes.png": "https://ydshieh-kosmos-2.hf.space/file=/home/user/app/images/six_planes.png",
}
for image_name, url in images.items():
    if not Path(image_name).exists():
        image = Image.open(requests.get(url, stream=True).raw)
        image.save(image_name)


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Image(label="Input image"),
            gr.Textbox(label="Prompt"),
            gr.Checkbox(label="Show bounding boxes", value=True),
        ],
        outputs=["image", "text"],
        examples=[
            ["snowman.png", "An image of"],
            ["two_dogs.jpg", "Describe this image in detail:"],
            ["six_planes.png", "What is going on?"],
        ],
        allow_flagging="never",
    )
    return demo
