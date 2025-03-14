
import os
import re
import tempfile
from collections.abc import Iterator
from threading import Thread

import cv2
import gradio as gr
import torch
from PIL import Image
from transformers import TextIteratorStreamer

MAX_NUM_IMAGES = int(os.getenv("MAX_NUM_IMAGES", "5"))


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
                "files": ["assets/additional-examples/barchart.png"],
            }
        ],
        [
            {
                "text": "What is odd about this video?",
                "files": ["assets/additional-examples/tmp.mp4"],
            }
        ],
        [
            {
                "text": "I already have this supplement <image> and I want to buy this one <image>. Any warnings I should know about?",
                "files": ["assets/additional-examples/pill1.png", "assets/additional-examples/pill2.png"],
            }
        ],
        [
            {
                "text": "Write a poem inspired by the visual elements of the images.",
                "files": ["assets/sample-images/06-1.png", "assets/sample-images/06-2.png"],
            }
        ],
        [
            {
                "text": "Compose a short musical piece inspired by the visual elements of the images.",
                "files": [
                    "assets/sample-images/07-1.png",
                    "assets/sample-images/07-2.png",
                    "assets/sample-images/07-3.png",
                    "assets/sample-images/07-4.png",
                ],
            }
        ],
        [
            {
                "text": "Write a short story about what might have happened in this house.",
                "files": ["assets/sample-images/08.png"],
            }
        ],
        [
            {
                "text": "Create a short story based on the sequence of images.",
                "files": [
                    "assets/sample-images/09-1.png",
                    "assets/sample-images/09-2.png",
                    "assets/sample-images/09-3.png",
                    "assets/sample-images/09-4.png",
                    "assets/sample-images/09-5.png",
                ],
            }
        ],
        [
            {
                "text": "Describe the creatures that would live in this world.",
                "files": ["assets/sample-images/10.png"],
            }
        ],
        [
            {
                "text": "Read text in the image.",
                "files": ["assets/additional-examples/1.png"],
            }
        ],
        [
            {
                "text": "When is this ticket dated and how much did it cost?",
                "files": ["assets/additional-examples/2.png"],
            }
        ],
        [
            {
                "text": "Read the text in the image into markdown.",
                "files": ["assets/additional-examples/3.png"],
            }
        ],
        [
            {
                "text": "Evaluate this integral.",
                "files": ["assets/additional-examples/4.png"],
            }
        ],
        [
            {
                "text": "caption this image",
                "files": ["assets/sample-images/01.png"],
            }
        ],
        [
            {
                "text": "What's the sign says?",
                "files": ["assets/sample-images/02.png"],
            }
        ],
        [
            {
                "text": "Compare and contrast the two images.",
                "files": ["assets/sample-images/03.png"],
            }
        ],
        [
            {
                "text": "List all the objects in the image and their colors.",
                "files": ["assets/sample-images/04.png"],
            }
        ],
        [
            {
                "text": "Describe the atmosphere of the scene.",
                "files": ["assets/sample-images/05.png"],
            }
        ],
    ]

    DESCRIPTION = """\
    <img src='https://huggingface.co/spaces/huggingface-projects/gemma-3-12b-it/resolve/main/assets/logo.png' id='logo' />
    This is a demo of Gemma 3, a vision language model with outstanding performance on a wide range of tasks.
    You can upload images, interleaved images and videos. Note that video input only supports single-turn conversation and mp4 input.
    """

    demo = gr.ChatInterface(
        fn=run,
        type="messages",
        chatbot=gr.Chatbot(type="messages", scale=1, allow_tags=["image"]),
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
