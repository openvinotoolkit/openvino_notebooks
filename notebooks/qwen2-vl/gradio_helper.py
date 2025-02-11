from pathlib import Path
import gradio as gr


from PIL import Image
import numpy as np
import requests
from threading import Event, Thread
import inspect
from queue import Queue

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
    if not Path(file_name).exists():
        Image.open(requests.get(url, stream=True).raw).save(file_name)


def make_demo(model):
    import openvino_genai
    import openvino as ov

    has_additonal_buttons = "undo_button" in inspect.signature(gr.ChatInterface.__init__).parameters

    def read_image(path: str) -> ov.Tensor:
        """

        Args:
            path: The path to the image.

        Returns: the ov.Tensor containing the image.

        """
        pic = Image.open(path).convert("RGB")
        image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.byte)
        return ov.Tensor(image_data)

    class TextQueue:
        def __init__(self) -> None:
            self.text_queue = Queue()
            self.stop_signal = None
            self.stop_tokens = []

        def __call__(self, text):
            self.text_queue.put(text)

        def __iter__(self):
            return self

        def __next__(self):
            value = self.text_queue.get()
            if value == self.stop_signal or value in self.stop_tokens:
                raise StopIteration()
            else:
                return value

        def reset(self):
            self.text_queue = Queue()

        def end(self):
            self.text_queue.put(self.stop_signal)

    def bot_streaming(message, history):
        print(f"message is - {message}")
        print(f"history is - {history}")

        if not history:
            model.start_chat()
        generation_config = openvino_genai.GenerationConfig()
        generation_config.max_new_tokens = 128
        files = message["files"] if isinstance(message, dict) else message.files
        message_text = message["text"] if isinstance(message, dict) else message.text

        image = None
        if files:
            # message["files"][-1] is a Dict or just a string
            if isinstance(files[-1], dict):
                image = files[-1]["path"]
            else:
                if isinstance(files[-1], (str, Path)):
                    image = files[-1]
                else:
                    image = files[-1] if isinstance(files[-1], (list, tuple)) else files[-1].path
        if image is not None:
            image = read_image(image)
        streamer = TextQueue()
        stream_complete = Event()

        def generate_and_signal_complete():
            """
            generation function for single thread
            """
            streamer.reset()
            generation_kwargs = {"prompt": message_text, "generation_config": generation_config, "streamer": streamer}
            if image is not None:
                generation_kwargs["image"] = image
            model.generate(**generation_kwargs)
            stream_complete.set()
            streamer.end()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            yield buffer

    additional_buttons = {}
    if has_additonal_buttons:
        additional_buttons = {"undo_button": None, "retry_button": None}
    demo = gr.ChatInterface(
        fn=bot_streaming,
        title="Qwen2-VL OpenVINO Demo",
        examples=[
            {"text": "What is on the flower?", "files": ["./bee.jpg"]},
            {"text": "How to make this pastry?", "files": ["./baklava.png"]},
        ],
        stop_btn=None,
        multimodal=True,
        **additional_buttons,
    )
    return demo
