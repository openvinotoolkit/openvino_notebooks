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
            try:
                Image.open(requests.get(url, stream=True, timeout=30).raw).save(file_name)
            except Exception:
                pass


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
        if item["role"] != "user":
            continue
        # Gradio 6: content is always a list of content blocks
        for block in item.get("content", []):
            if not isinstance(block, dict) or block.get("type") != "file":
                continue
            file_path = block.get("file", {}).get("path", "")
            if file_path.endswith(".mp4"):
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


def process_history(history: list[dict]) -> tuple[list[dict], list[Image.Image]]:
    """Convert Gradio chat history to HF messages format and collect images."""
    messages = []
    images = []
    current_user_content: list[dict] = []
    for item in history:
        if item["role"] == "assistant":
            if current_user_content:
                messages.append({"role": "user", "content": current_user_content})
                current_user_content = []
            # Gradio 6: assistant content is a list of blocks
            text = ""
            for block in item.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
                elif isinstance(block, str):
                    text += block
            messages.append({"role": "assistant", "content": text})
        else:
            # Gradio 6: user content is a list of blocks
            for block in item.get("content", []):
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        current_user_content.append({"type": "text", "text": block.get("text", "")})
                    elif block.get("type") == "file":
                        file_path = block.get("file", {}).get("path", "")
                        if not file_path.endswith(".mp4"):
                            pic = Image.open(file_path).convert("RGB")
                            images.append(pic)
                            current_user_content.append({"type": "image", "image": pic})
                elif isinstance(block, str):
                    current_user_content.append({"type": "text", "text": block})
    if current_user_content:
        messages.append({"role": "user", "content": current_user_content})
    return messages, images


def make_demo(model, processor):
    download_example_images()

    def run(message: dict, history: list[dict], system_prompt: str = "", max_new_tokens: int = 512, enable_thinking: bool = False) -> Iterator[str]:
        if not validate_media_constraints(message, history):
            yield ""
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        history_messages, history_images = process_history(history)
        messages.extend(history_messages)

        # Build current user message content (images before text for best quality)
        user_content = []
        current_images = []
        if message["files"]:
            for file_path in message["files"]:
                if file_path.endswith(".mp4"):
                    # Process video frames as images
                    frames = downsample_video(file_path)
                    for pil_image, timestamp in frames:
                        current_images.append(pil_image)
                        user_content.append({"type": "text", "text": f"Frame {timestamp}:"})
                        user_content.append({"type": "image", "image": pil_image})
                else:
                    pic = Image.open(file_path).convert("RGB")
                    current_images.append(pic)
                    user_content.append({"type": "image", "image": pic})
        user_content.append({"type": "text", "text": message["text"]})
        messages.append({"role": "user", "content": user_content})

        all_images = history_images + current_images

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)

        # Process inputs
        if all_images:
            inputs = processor(text=text, images=all_images, return_tensors="pt")
        else:
            inputs = processor(text=text, return_tensors="pt")

        # Streaming generation
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer)
        Thread(target=model.generate, kwargs=generation_kwargs).start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            yield buffer

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
    This is a demo of **Gemma 4** with OpenVINO — a multimodal model supporting text, images, and video.
    Upload images, use interleaved `<image>` tags, or attach an mp4 video (single-turn only).
    Enable **Thinking mode** to let the model reason step-by-step before answering.
    """

    demo = gr.ChatInterface(
        fn=run,
        chatbot=gr.Chatbot(scale=1),
        textbox=gr.MultimodalTextbox(file_types=["image", ".mp4"], file_count="multiple", autofocus=True),
        multimodal=True,
        additional_inputs=[
            gr.Textbox(label="System Prompt", value="You are a helpful assistant."),
            gr.Slider(label="Max New Tokens", minimum=100, maximum=2000, step=10, value=700),
            gr.Checkbox(label="Enable Thinking Mode", value=False),
        ],
        stop_btn=False,
        title="Gemma 4 with OpenVINO",
        description=DESCRIPTION,
        examples=examples,
        run_examples_on_click=False,
        cache_examples=False,
    )

    return demo
