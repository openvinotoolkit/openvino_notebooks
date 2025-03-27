import gradio as gr
from transformers import TextIteratorStreamer
from transformers.image_utils import load_image
from threading import Thread
from pathlib import Path
import re
import ast
import html
import random

from PIL import ImageOps

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument


def get_examples():
    example_images = {
        "ui.png": "https://github.com/user-attachments/assets/2118de40-a389-49a7-96af-e92c6a921cbb",
        "gazette_de_france.jpg": "https://github.com/user-attachments/assets/2fb32e84-d972-4056-b047-2a081e909a3b",
        "redhat.png": "https://github.com/user-attachments/assets/7805b559-fdc5-4030-9b28-513ba7df2d73",
        "paper.png": "https://github.com/user-attachments/assets/a39a245c-7309-407f-ade1-3e0e34aea9c7",
        "table.jpg": "https://github.com/user-attachments/assets/5df66c16-0410-41c2-8fc0-0088ab4a2d3a",
        "manual.png": "https://github.com/user-attachments/assets/a67c9305-a3ca-42ac-837b-59056f1ec382",
        "code.png": "https://github.com/user-attachments/assets/5a9614b8-7e57-4a39-8723-e2f669873ff3",
        "formula.jpg": "https://github.com/user-attachments/assets/3d1dfd7f-6ef1-4a3d-994f-0195b907b6fb",
        "chart.png": "https://github.com/user-attachments/assets/7a76e41d-bab6-4ca7-a93f-a17ef2054e0b",
    }
    for file_name, url in example_images.items():
        if not Path(file_name).exists():
            load_image(url).save(file_name)


def add_random_padding(image, min_percent=0.1, max_percent=0.10):
    image = image.convert("RGB")

    width, height = image.size

    pad_w_percent = random.uniform(min_percent, max_percent)
    pad_h_percent = random.uniform(min_percent, max_percent)

    pad_w = int(width * pad_w_percent)
    pad_h = int(height * pad_h_percent)

    corner_pixel = image.getpixel((0, 0))  # Top-left corner
    padded_image = ImageOps.expand(image, border=(pad_w, pad_h, pad_w, pad_h), fill=corner_pixel)

    return padded_image


def normalize_values(text, target_max=500):
    def normalize_list(values):
        max_value = max(values) if values else 1
        return [round((v / max_value) * target_max) for v in values]

    def process_match(match):
        num_list = ast.literal_eval(match.group(0))
        normalized = normalize_list(num_list)
        return "".join([f"<loc_{num}>" for num in normalized])

    pattern = r"\[([\d\.\s,]+)\]"
    normalized_text = re.sub(pattern, process_match, text)
    return normalized_text


def make_demo(model, processor):
    get_examples()

    def model_inference(input_dict, history):
        text = input_dict["text"]
        print(input_dict["files"])
        if len(input_dict["files"]) > 1:
            if "OTSL" in text or "code" in text:
                images = [add_random_padding(load_image(image)) for image in input_dict["files"]]
            else:
                images = [load_image(image) for image in input_dict["files"]]

        elif len(input_dict["files"]) == 1:
            if "OTSL" in text or "code" in text:
                images = [add_random_padding(load_image(input_dict["files"][0]))]
            else:
                images = [load_image(input_dict["files"][0])]

        else:
            images = []

        if text == "" and not images:
            gr.Error("Please input a query and optionally image(s).")

        if text == "" and images:
            gr.Error("Please input a text query along the image(s).")

        if "OCR at text at" in text or "Identify element" in text or "formula" in text:
            text = normalize_values(text, target_max=500)

        resulting_messages = [{"role": "user", "content": [{"type": "image"} for _ in range(len(images))] + [{"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(resulting_messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[images], return_tensors="pt")

        generation_args = {
            "input_ids": inputs.input_ids,
            "pixel_values": inputs.pixel_values,
            "attention_mask": inputs.attention_mask,
            "num_return_sequences": 1,
            "max_new_tokens": 8192,
            "eos_token_id": [49279, 2],
        }

        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=False)
        generation_args = dict(inputs, streamer=streamer, max_new_tokens=8192)

        thread = Thread(target=model.generate, kwargs=generation_args)
        thread.start()

        yield "..."
        buffer = ""
        full_output = ""

        for new_text in streamer:
            full_output += new_text
            buffer += html.escape(new_text)
            yield buffer

        cleaned_output = full_output.replace("<end_of_utterance>", "").strip()

        if cleaned_output:
            doctag_output = cleaned_output
            yield cleaned_output

        if any(tag in doctag_output for tag in ["<doctag>", "<otsl>", "<code>", "<chart>", "<formula>"]):
            doc = DoclingDocument(name="Document")
            if "<chart>" in doctag_output:
                doctag_output = doctag_output.replace("<chart>", "<otsl>").replace("</chart>", "</otsl>")
                doctag_output = re.sub(r"(<loc_500>)(?!.*<loc_500>)<[^>]+>", r"\1", doctag_output)

            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctag_output], images)
            doc.load_from_doctags(doctags_doc)
            yield f"**MD Output:**\n\n{doc.export_to_markdown()}"

    examples = [
        [{"text": "Convert this page to docling.", "files": ["manual.png"]}],
        [{"text": "Convert this table to OTSL.", "files": ["table.jpg"]}],
        [{"text": "Convert code to text.", "files": ["code.png"]}],
        [{"text": "Convert formula to latex.", "files": ["formula.jpg"]}],
        [{"text": "OCR the text in location [47, 531, 167, 565]", "files": ["ui.png"]}],
        [{"text": "Extract all section header elements on the page.", "files": ["paper.png"]}],
        [{"text": "Identify element at location [123, 413, 1059, 1061]", "files": ["redhat.png"]}],
        [{"text": "Convert this page to docling.", "files": ["gazette_de_france.jpg"]}],
    ]

    demo = gr.ChatInterface(
        fn=model_inference,
        title="SmolDocling-256M: Ultra-compact VLM for Document Conversion 💫",
        description="Play with [ds4sd/SmolDocling-256M-preview](https://huggingface.co/ds4sd/SmolDocling-256M-preview) in this demo. To get started, upload an image and text or try one of the examples. This demo doesn't use history for the chat, so every chat you start is a new conversation.",
        examples=examples,
        textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image"], file_count="multiple"),
        stop_btn="Stop Generation",
        multimodal=True,
        cache_examples=False,
    )

    return demo
