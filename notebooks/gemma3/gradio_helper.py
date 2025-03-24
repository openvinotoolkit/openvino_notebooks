import os
import re
import tempfile
from collections.abc import Iterator
from threading import Thread

from pathlib import Path
import cv2
import gradio as gr
import requests
from PIL import Image
from transformers import TextIteratorStreamer

MAX_NUM_IMAGES = int(os.getenv("MAX_NUM_IMAGES", "5"))

example_images = {
    "barchart.png": "https://github.com/user-attachments/assets/7779e110-691a-40db-b7db-f226cd4d06bd",
    "sunset.png": "https://github.com/user-attachments/assets/da3edb79-ae36-4973-9eaf-6ef712425faa",
    "colors.png": "https://github.com/user-attachments/assets/d8e027f5-27d9-4d4d-9195-e89f8b972cb0",
    "sign.png": "https://github.com/user-attachments/assets/491c4af5-dc55-477b-9dc0-0960742980f2",
    "integral.png": "https://github.com/user-attachments/assets/8e9662f2-01fe-485d-8110-b5ce2d0d2b27",
    "house.png": "https://github.com/user-attachments/assets/a395f740-6e9a-4fa7-823b-e2862b910891",
}


def download_example_images():
    for file_name, url in example_images.items():
        if not Path(file_name).exists():
            Image.open(requests.get(url, stream=True).raw).save(file_name)


def count_files_in_new_message(paths: list[str]) -> tuple[int, int]:
    image_count = 0
    video_count = 0
    for path in paths:
        if path.endswith(".mp4"):
            video_count += 1
        else:
            image_count += 1
    return image_count, video_count


def count_files_in_history(history: list[dict]) -> tuple[int, int]:
    image_count = 0
    video_count = 0
    for item in history:
        if item["role"] != "user" or isinstance(item["content"], str):
            continue
        if item["content"][0].endswith(".mp4"):
            video_count += 1
        else:
            image_count += 1
    return image_count, video_count


def validate_media_constraints(message: dict, history: list[dict]) -> bool:
    new_image_count, new_video_count = count_files_in_new_message(message["files"])
    history_image_count, history_video_count = count_files_in_history(history)
    image_count = history_image_count + new_image_count
    video_count = history_video_count + new_video_count
    if video_count > 1:
        gr.Warning("Only one video is supported.")
        return False
    if video_count == 1:
        if image_count > 0:
            gr.Warning("Mixing images and videos is not allowed.")
            return False
        if "<image>" in message["text"]:
            gr.Warning("Using <image> tags with video files is not supported.")
            return False
        # TODO: Add frame count validation for videos similar to image count limits  # noqa: FIX002, TD002, TD003
    if video_count == 0 and image_count > MAX_NUM_IMAGES:
        gr.Warning(f"You can upload up to {MAX_NUM_IMAGES} images.")
        return False
    if "<image>" in message["text"] and message["text"].count("<image>") != new_image_count:
        gr.Warning("The number of <image> tags in the text does not match the number of images.")
        return False
    return True


def downsample_video(video_path: str) -> list[tuple[Image.Image, float]]:
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = int(fps / 3)
    frames = []

    for i in range(0, total_frames, frame_interval):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))

    vidcap.release()
    return frames


def process_video(video_path: str) -> list[dict]:
    content = []
    frames = downsample_video(video_path)
    for frame in frames:
        pil_image, timestamp = frame
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            pil_image.save(temp_file.name)
            content.append({"type": "text", "text": f"Frame {timestamp}:"})
            content.append({"type": "image", "url": temp_file.name})
    return content


def process_interleaved_images(message: dict) -> list[dict]:
    parts = re.split(r"(<image>)", message["text"])

    content = []
    image_index = 0
    for part in parts:
        if part == "<image>":
            content.append({"type": "image", "url": message["files"][image_index]})
            image_index += 1
        elif part.strip():
            content.append({"type": "text", "text": part.strip()})
        elif isinstance(part, str) and part != "<image>":
            content.append({"type": "text", "text": part})
    return content


def process_new_user_message(message: dict) -> list[dict]:
    if not message["files"]:
        return [{"type": "text", "text": message["text"]}]

    if message["files"][0].endswith(".mp4"):
        return [{"type": "text", "text": message["text"]}, *process_video(message["files"][0])]

    if "<image>" in message["text"]:
        return process_interleaved_images(message)

    return [
        {"type": "text", "text": message["text"]},
        *[{"type": "image", "url": path} for path in message["files"]],
    ]


def process_history(history: list[dict]) -> list[dict]:
    messages = []
    current_user_content: list[dict] = []
    for item in history:
        if item["role"] == "assistant":
            if current_user_content:
                messages.append({"role": "user", "content": current_user_content})
                current_user_content = []
            messages.append({"role": "assistant", "content": [{"type": "text", "text": item["content"]}]})
        else:
            content = item["content"]
            if isinstance(content, str):
                current_user_content.append({"type": "text", "text": content})
            else:
                current_user_content.append({"type": "image", "url": content[0]})
    return messages


def make_demo(model, processor):
    download_example_images()

    def run(message: dict, history: list[dict], system_prompt: str = "", max_new_tokens: int = 512) -> Iterator[str]:
        if not validate_media_constraints(message, history):
            yield ""
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.extend(process_history(history))
        messages.append({"role": "user", "content": process_new_user_message(message)})

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device=model.device)

        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
        )
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        output = ""
        for delta in streamer:
            output += delta
            yield output

    examples = [
        [
            {
                "text": "I need to be in Japan for 10 days, going to Tokyo, Kyoto and Osaka. Think about number of attractions in each of them and allocate number of days to each city. Make public transport recommendations.",
                "files": [],
            }
        ],
        [
            {
                "text": "Write the matplotlib code to generate the same bar chart.",
                "files": ["barchart.png"],
            }
        ],
        [
            {
                "text": "Write a short story about what might have happened in this house.",
                "files": ["house.png"],
            }
        ],
        [
            {
                "text": "Evaluate this integral.",
                "files": ["integral.png"],
            }
        ],
        [
            {
                "text": "What's the sign says?",
                "files": ["sign.png"],
            }
        ],
        [
            {
                "text": "List all the objects in the image and their colors.",
                "files": ["colors.png"],
            }
        ],
        [
            {
                "text": "Describe the atmosphere of the scene.",
                "files": ["sunset.png"],
            }
        ],
    ]

    DESCRIPTION = """\
    This is a demo of Gemma 3, a vision language model with outstanding performance on a wide range of tasks.
    You can upload images, interleaved images and videos. Note that video input only supports single-turn conversation and mp4 input.
    """

    demo = gr.ChatInterface(
        fn=run,
        type="messages",
        chatbot=gr.Chatbot(type="messages", scale=1),
        textbox=gr.MultimodalTextbox(file_types=["image", ".mp4"], file_count="multiple", autofocus=True),
        multimodal=True,
        additional_inputs=[
            gr.Textbox(label="System Prompt", value="You are a helpful assistant."),
            gr.Slider(label="Max New Tokens", minimum=100, maximum=2000, step=10, value=700),
        ],
        stop_btn=False,
        title="Gemma 3 OpenVINO",
        description=DESCRIPTION,
        examples=examples,
        run_examples_on_click=False,
        cache_examples=False,
        delete_cache=(1800, 1800),
    )

    return demo
