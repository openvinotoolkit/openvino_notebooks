from pathlib import Path
from threading import Thread

import gradio as gr
import requests
from PIL import Image
from transformers import TextIteratorStreamer


MAX_IMAGE_SIDE = 512


def download_examples():
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
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            Image.open(response.raw).save(file_name)


def get_message_value(message, key, default=None):
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def get_file_paths(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    if isinstance(content, dict):
        if content.get("path"):
            return [content["path"]]
        nested_items = [*content.get("files", []), *content.get("content", [])]
        return [path for item in nested_items for path in get_file_paths(item)]
    if isinstance(content, (list, tuple)):
        return [path for item in content for path in get_file_paths(item)]

    path = getattr(content, "path", None)
    return [path] if path else []


def get_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max(image.size) > MAX_IMAGE_SIDE:
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        return image


def history_to_turns(history):
    turns = []
    pending_text = []
    pending_files = []

    for entry in history:
        if not isinstance(entry, dict):
            continue

        role = entry.get("role")
        content = entry.get("content")
        if role == "user":
            if not isinstance(content, str):
                pending_files.extend(get_file_paths(content))
            text = get_text(content)
            if text:
                pending_text.append(text)
        elif role == "assistant" and pending_text:
            turns.append(
                {
                    "text": " ".join(pending_text),
                    "files": pending_files,
                    "answer": get_text(content),
                }
            )
            pending_text = []
            pending_files = []

    return turns, pending_text, pending_files


def make_demo(model, processor):
    download_examples()

    def bot_streaming(message, history):
        message_text = (get_message_value(message, "text", "") or "").strip()
        if not message_text:
            raise gr.Error("Please provide a text question.")

        turns, pending_text, pending_files = history_to_turns(history)
        current_files = pending_files + get_file_paths(get_message_value(message, "files", []))
        current_files = list(dict.fromkeys(current_files))
        if pending_text:
            message_text = " ".join([*pending_text, message_text])

        turns.append({"text": message_text, "files": current_files, "answer": None})

        messages = []
        images = []
        for turn in turns:
            turn_images = [load_image(path) for path in turn["files"]]
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in turn_images]
                    + [{"type": "text", "text": turn["text"]}],
                }
            )
            images.extend(turn_images)
            if turn["answer"]:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": turn["answer"]}],
                    }
                )

        if not images:
            raise gr.Error("Upload an image in the first message to start a conversation.")

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=images, text=prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_errors = []

        def generate():
            try:
                model.generate(**inputs, streamer=streamer, max_new_tokens=128, do_sample=False)
            except Exception as error:
                generation_errors.append(error)
                streamer.on_finalized_text("", stream_end=True)

        thread = Thread(target=generate, daemon=True)
        thread.start()

        answer = ""
        for new_text in streamer:
            answer += new_text
            yield answer

        thread.join()
        if generation_errors:
            raise gr.Error(f"Generation failed: {generation_errors[0]}")

    return gr.ChatInterface(
        fn=bot_streaming,
        type="messages",
        title="Ministral-3 OpenVINO Demo",
        description="Upload one or more images and continue the conversation in subsequent turns.",
        examples=[
            [{"text": "What is on the flower?", "files": ["./bee.jpg"]}],
            [{"text": "How to make this pastry?", "files": ["./baklava.png"]}],
        ],
        textbox=gr.MultimodalTextbox(label="Message", file_types=["image"], file_count="multiple"),
        stop_btn=None,
        multimodal=True,
    )
