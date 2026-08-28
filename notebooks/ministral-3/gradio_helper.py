from pathlib import Path
from threading import Thread

import gradio as gr
import requests
from PIL import Image
from transformers import TextIteratorStreamer

MAX_IMAGE_SIDE = 512
THINK_START = "[THINK]"
THINK_END = "[/THINK]"
DETAILS_PREFIX = "<details><summary>Reasoning</summary>\n\n"
DETAILS_OPEN_PREFIX = "<details open><summary>Reasoning</summary>\n\n"
DETAILS_SUFFIX = "\n\n</details>"


def strip_terminal_tokens(text, tokenizer):
    for token in (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token):
        if token:
            text = text.replace(token, "")
    return text.strip()


def split_reasoning_output(text, tokenizer, final=False):
    end_pos = text.find(THINK_END)
    if end_pos >= 0:
        start_pos = text.find(THINK_START)
        reasoning_start = start_pos + len(THINK_START) if start_pos >= 0 else 0
        reasoning = strip_terminal_tokens(text[reasoning_start:end_pos], tokenizer)
        answer = strip_terminal_tokens(text[end_pos + len(THINK_END) :], tokenizer)
        return reasoning, answer, True

    text = text.replace(THINK_START, "", 1)
    if final:
        return None, strip_terminal_tokens(text, tokenizer), True
    return strip_terminal_tokens(text, tokenizer), "", False


def format_reasoning_response(reasoning, answer, complete):
    if not reasoning:
        return answer
    prefix = DETAILS_PREFIX if complete else DETAILS_OPEN_PREFIX
    response = f"{prefix}{reasoning}{DETAILS_SUFFIX}"
    return f"{response}\n\n{answer}" if answer else response


def parse_display_response(text):
    for prefix in (DETAILS_PREFIX, DETAILS_OPEN_PREFIX):
        if text.startswith(prefix) and DETAILS_SUFFIX in text:
            reasoning, answer = text[len(prefix) :].split(DETAILS_SUFFIX, 1)
            return reasoning.strip(), answer.strip()
    return None, text.strip()


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
        return " ".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
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
            reasoning, answer = parse_display_response(get_text(content))
            turns.append(
                {
                    "text": " ".join(pending_text),
                    "files": pending_files,
                    "reasoning": reasoning,
                    "answer": answer,
                }
            )
            pending_text = []
            pending_files = []

    return turns, pending_text, pending_files


def make_demo(model, processor, is_reasoning_model=False):
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

        turns.append({"text": message_text, "files": current_files, "reasoning": None, "answer": None})

        messages = []
        images = []
        for turn in turns:
            turn_images = [load_image(path) for path in turn["files"]]
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in turn_images] + [{"type": "text", "text": turn["text"]}],
                }
            )
            images.extend(turn_images)
            if turn["reasoning"] or turn["answer"]:
                assistant_content = []
                if turn["reasoning"]:
                    assistant_content.append({"type": "thinking", "thinking": turn["reasoning"], "closed": True})
                if turn["answer"]:
                    assistant_content.append({"type": "text", "text": turn["answer"]})
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    }
                )

        if not images:
            raise gr.Error("Upload an image in the first message to start a conversation.")

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=images, text=prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=not is_reasoning_model,
        )
        generation_errors = []

        def generate():
            try:
                generation_kwargs = (
                    {"max_new_tokens": 1024, "do_sample": True, "temperature": 0.7, "top_p": 0.95}
                    if is_reasoning_model
                    else {"max_new_tokens": 128, "do_sample": False}
                )
                model.generate(**inputs, streamer=streamer, **generation_kwargs)
            except Exception as error:
                generation_errors.append(error)
                streamer.on_finalized_text("", stream_end=True)

        thread = Thread(target=generate, daemon=True)
        thread.start()

        raw_output = ""
        last_response = ""
        for new_text in streamer:
            raw_output += new_text
            if is_reasoning_model:
                reasoning, answer, complete = split_reasoning_output(raw_output, processor.tokenizer)
                last_response = format_reasoning_response(reasoning, answer, complete)
            else:
                last_response = raw_output
            yield last_response

        thread.join()
        if generation_errors:
            raise gr.Error(f"Generation failed: {generation_errors[0]}")
        if is_reasoning_model:
            reasoning, answer, complete = split_reasoning_output(raw_output, processor.tokenizer, final=True)
            final_response = format_reasoning_response(reasoning, answer, complete)
            if final_response != last_response:
                yield final_response

    return gr.ChatInterface(
        fn=bot_streaming,
        type="messages",
        title="Ministral-3 OpenVINO Demo",
        description=(
            "Upload one or more images and continue the conversation in subsequent turns. "
            "Reasoning traces are shown in a collapsible section and preserved in the model context."
        ),
        examples=[
            [{"text": "What is on the flower?", "files": ["./bee.jpg"]}],
            [{"text": "How to make this pastry?", "files": ["./baklava.png"]}],
        ],
        textbox=gr.MultimodalTextbox(label="Message", file_types=["image"], file_count="multiple"),
        stop_btn=None,
        multimodal=True,
    )
