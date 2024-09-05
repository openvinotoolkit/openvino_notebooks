from PIL import Image
from typing import Callable
import gradio as gr
import requests

example_image_urls = [
    (
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1d6a0188-5613-418d-a1fd-4560aae1d907",
        "bee.jpg",
    ),
    (
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/6cc7feeb-0721-4b5d-8791-2576ed9d2863",
        "baklava.png",
    ),
]
for url, file_name in example_image_urls:
    Image.open(requests.get(url, stream=True).raw).save(file_name)


def make_demo(fn: Callable):
    demo = gr.ChatInterface(
        fn=fn,
        title="LLaVA NeXT",
        description="Try [LLaVA NeXT](https://huggingface.co/docs/transformers/main/en/model_doc/llava_next) in this demo using OpenVINO. Upload an image and start chatting about it, or simply try one of the examples below. If you don't upload an image, you will receive an error.",
        examples=[
            {"text": "What is on the flower?", "files": ["./bee.jpg"]},
            {"text": "How to make this pastry?", "files": ["./baklava.png"]},
        ],
        stop_btn="Stop Generation",
        multimodal=True,
    )
    return demo
