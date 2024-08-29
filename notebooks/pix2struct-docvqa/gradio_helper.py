from pathlib import Path
from typing import Callable
import gradio as gr
import requests
from PIL import Image


example_images_urls = [
    "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/94ef687c-aebb-452b-93fe-c7f29ce19503",
    "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/70b2271c-9295-493b-8a5c-2f2027dcb653",
    "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1e2be134-0d45-4878-8e6c-08cfc9c8ea3d",
]

file_names = ["eiffel_tower.png", "exsibition.jpeg", "population_table.jpeg"]

for img_url, image_file in zip(example_images_urls, file_names):
    if not Path(image_file).exists():
        Image.open(requests.get(img_url, stream=True).raw).save(image_file)

questions = [
    "What is Eiffel tower tall?",
    "When is the coffee break?",
    "What the population of Stoddard?",
]

examples = [list(pair) for pair in zip(file_names, questions)]


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=["image", "text"],
        outputs="text",
        title="Pix2Struct for DocVQA",
        examples=examples,
        cache_examples=False,
        allow_flagging="never",
    )
    return demo
