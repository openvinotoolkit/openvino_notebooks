from pathlib import Path
from typing import Callable
import gradio as gr
import requests
from PIL import Image


example_image_urls = [
    (
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1d6a0188-5613-418d-a1fd-4560aae1d907",
        "bee.jpg",
    ),
    (
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/6cc7feeb-0721-4b5d-8791-2576ed9d2863",
        "baklava.png",
    ),
    ("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/dd5105d6-6a64-4935-8a34-3058a82c8d5d", "small.png"),
    ("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1221e2a8-a6da-413a-9af6-f04d56af3754", "chart.png"),
]
for url, file_name in example_image_urls:
    if not Path(file_name).exists():
        Image.open(requests.get(url, stream=True).raw).save(file_name)


def make_demo(fn: Callable):
    demo = gr.ChatInterface(
        fn=fn,
        title="🚀nanoLLaVA",
        examples=[
            {"text": "What is on the flower?", "files": ["./bee.jpg"]},
            {"text": "How to make this pastry?", "files": ["./baklava.png"]},
            {"text": "What is the text saying?", "files": ["./small.png"]},
            {"text": "What does the chart display?", "files": ["./chart.png"]},
        ],
        description="Try [nanoLLaVA](https://huggingface.co/qnguyen3/nanoLLaVA) using OpenVINO in this demo. Upload an image and start chatting about it, or simply try one of the examples below. If you don't upload an image, you will receive an error.",
        stop_btn="Stop Generation",
        multimodal=True,
    )
    return demo
