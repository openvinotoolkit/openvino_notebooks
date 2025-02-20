from pathlib import Path
import requests
import gradio as gr
from PIL import Image
from threading import Event, Thread
from transformers import TextIteratorStreamer
from queue import Queue

import openvino_genai as ov_genai


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


def make_demo(pipe, processor, read_images, model_name):
    example_image_urls = [
        ("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/dd5105d6-6a64-4935-8a34-3058a82c8d5d", "small.png"),
        ("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1221e2a8-a6da-413a-9af6-f04d56af3754", "chart.png"),
    ]

    for url, file_name in example_image_urls:
        if not Path(file_name).exists():
            Image.open(requests.get(url, stream=True).raw).save(file_name)

    def bot_streaming(message, history):
        print(f"message is - {message}")
        print(f"history is - {history}")
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
        try:
            if image is None:
                # Handle the case where image is None
                raise gr.Error("You need to upload an image for Phi3-Vision to work. Close the error and try again with an Image.")
        except NameError:
            # Handle the case where 'image' is not defined at all
            raise gr.Error("You need to upload an image for Phi3-Vision to work. Close the error and try again with an Image.")
        image = read_images(image)

        conversation = []
        flag = False
        for user, assistant in history:
            if assistant is None:
                # pass
                flag = True
                conversation.extend([{"role": "user", "content": ""}])
                continue
            if flag == True:
                conversation[0]["content"] = f"<|image_1|>\n{user}"
                conversation.extend([{"role": "assistant", "content": assistant}])
                flag = False
                continue
            conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        if len(history) == 0:
            conversation.append({"role": "user", "content": f"<|image_1|>\n{message_text}"})
        else:
            conversation.append({"role": "user", "content": message_text})
        print(f"prompt is -\n{conversation}")

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 200
        config.do_sample = False
        config.set_eos_token_id(pipe.get_tokenizer().get_eos_token_id())
        prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        streamer = TextIteratorStreamer(processor)
        streamer = TextQueue()
        stream_complete = Event()

        generation_kwargs = dict(prompt=prompt, streamer=streamer, generation_config=config)

        def generate_and_signal_complete():
            """
            generation function for single thread
            """
            streamer.reset()
            generation_kwargs = {"prompt": prompt, "generation_config": config, "streamer": streamer}
            if image is not None:
                generation_kwargs["images"] = image
            pipe.generate(**generation_kwargs)
            stream_complete.set()
            streamer.end()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            yield buffer

    demo = gr.ChatInterface(
        fn=bot_streaming,
        title=f"{model_name} with OpenVINO",
        examples=[
            {"text": "What is the text saying?", "files": ["./small.png"]},
            {"text": "What does the chart display?", "files": ["./chart.png"]},
        ],
        description=f"Try the [{model_name} model](https://huggingface.co/microsoft/{model_name}) from Microsoft with OpenVINO. Upload an image and start chatting about it, or simply try one of the examples below. If you won't upload an image, you will receive an error.",
        stop_btn="Stop Generation",
        multimodal=True,
    )

    return demo
