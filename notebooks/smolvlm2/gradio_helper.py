import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread
import re
import time
import requests
from pathlib import Path
from PIL import Image


def download_examples():
    example_images = {
        "weather.png": "https://github.com/user-attachments/assets/85af4410-6e46-484d-b13b-fd9260eb2b7c",
        "newyork.jpg": "https://github.com/user-attachments/assets/c530b689-2ff6-4c4d-91bc-e6ac5331df59",
        "document.jpg": "https://github.com/user-attachments/assets/ac7225b6-bf90-4faf-b05f-bbba41a87142",
        "rococo.jpg": "https://github.com/user-attachments/assets/9e26e36e-f2be-4fa2-affd-891448abcc7d",
        "rococo_1.jpg": "https://github.com/user-attachments/assets/d39bdb95-833c-4ebd-8390-15a8fc2cd0b6",
    }
    for file_name, url in example_images.items():
        if not Path(file_name).exists():
            Image.open(requests.get(url, stream=True).raw).save(file_name)


def make_demo(model, processor):
    download_examples()

    def model_inference(input_dict, history, max_tokens):
        resulting_messages = []
        user_content = []
        media_queue = []
        for hist in history:
            if hist["role"] == "user" and isinstance(hist["content"], tuple):
                file_name = hist["content"][0]
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                media_queue.append({"type": "image", "path": file_name})
            elif file_name.endswith(".mp4"):
                media_queue.append({"type": "video", "path": file_name})

        for hist in history:
            if hist["role"] == "user" and isinstance(hist["content"], str):
                text = hist["content"]
                parts = re.split(r"(<image>|<video>)", text)

                for part in parts:
                    if part == "<image>" and media_queue:
                        user_content.append(media_queue.pop(0))
                    elif part == "<video>" and media_queue:
                        user_content.append(media_queue.pop(0))
                    elif part.strip():
                        user_content.append({"type": "text", "text": part.strip()})

            elif hist["role"] == "assistant":
                resulting_messages.append({"role": "user", "content": user_content})
                resulting_messages.append({"role": "assistant", "content": [{"type": "text", "text": hist["content"]}]})
                user_content = []

        text = input_dict["text"]
        c_user_content = []
        c_media_queue = []
        text = input_dict["text"].strip()
        for file in input_dict.get("files", []):
            if file.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                c_media_queue.append({"type": "image", "path": file})
            elif file.endswith((".mp4", ".mov", ".avi", ".mkv", ".flv")):
                c_media_queue.append({"type": "video", "path": file})

        if "<image>" in text or "<video>" in text:
            parts = re.split(r"(<image>|<video>)", text)
            for part in parts:
                if part == "<image>" and c_media_queue:
                    c_user_content.append(c_media_queue.pop(0))
                elif part == "<video>" and c_media_queue:
                    c_user_content.append(c_media_queue.pop(0))
                elif part.strip():
                    c_user_content.append({"type": "text", "text": part.strip()})
        else:
            c_user_content.append({"type": "text", "text": text})

            for media in c_media_queue:
                c_user_content.append(media)

        current_message = {"role": "user", "content": c_user_content}

        if text == "":
            gr.Error("Please input a query and optionally image(s).")
        resulting_messages.append(current_message)
        print("resulting_messages", resulting_messages)
        inputs = processor.apply_chat_template(
            resulting_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Generate
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generation_args = dict(inputs, streamer=streamer, max_new_tokens=max_tokens)
        generated_text = ""

        thread = Thread(target=model.generate, kwargs=generation_args)
        thread.start()

        yield "..."
        buffer = ""

        for new_text in streamer:
            buffer += new_text
            generated_text_without_prompt = buffer
            time.sleep(0.01)
            yield buffer

    examples = [
        [{"text": "Where do the severe droughts happen according to this diagram?", "files": ["weather.png"]}],
        [{"text": "What art era this artpiece <image> and this artpiece <image> belong to?", "files": ["rococo.jpg", "rococo_1.jpg"]}],
        [{"text": "Describe this image.", "files": ["newyork.jpg"]}],
        [{"text": "What is the date in this document?", "files": ["document.jpg"]}],
        [{"text": "What is happening in the video?", "files": ["dog.mp4"]}],
    ]
    demo = gr.ChatInterface(
        fn=model_inference,
        title="SmolVLM2: The Smollest Video Model Ever 📺",
        description="Play with SmolVLM2 and OpenVINO in this demo. To get started, upload an image and text or try one of the examples.",
        examples=examples,
        textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image", ".mp4"], file_count="multiple"),
        stop_btn="Stop Generation",
        multimodal=True,
        cache_examples=False,
        additional_inputs=[gr.Slider(minimum=100, maximum=500, step=50, value=200, label="Max Tokens")],
        type="messages",
    )

    return demo
