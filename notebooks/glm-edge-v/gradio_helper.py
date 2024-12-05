from pathlib import Path
import requests
import gradio as gr
import torch
from PIL import Image
from threading import Thread
from transformers import TextIteratorStreamer


def make_demo(model, processor, tokenizer):
    model_name = Path(model.config._name_or_path).parent.name

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
        if files:
            # message["files"][-1] is a Dict or just a string
            if isinstance(files[-1], dict):
                image = files[-1]["path"]
            else:
                image = files[-1] if isinstance(files[-1], (list, tuple)) else files[-1].path
        else:
            # if there's no image uploaded for this turn, look for images in the past turns
            # kept inside tuples, take the last one
            for hist in history:
                if type(hist[0]) == tuple:
                    image = hist[0][0]
        try:
            if image is None:
                # Handle the case where image is None
                raise gr.Error("You need to upload an image for GLM-Edeg-V to work. Close the error and try again with an Image.")
        except NameError:
            # Handle the case where 'image' is not defined at all
            raise gr.Error("You need to upload an image for GLM-Edeg-V to work. Close the error and try again with an Image.")

        conversation = []
        flag = False
        for user, assistant in history:
            if assistant is None:
                # pass
                flag = True
                conversation.extend([{"role": "user", "content": ""}])
                continue
            if flag == True:
                conversation[0]["content"] = [{"type": "image"}, {"type": "text", "text": user}]
                conversation.extend([{"role": "assistant", "content": [{"type": "text", "text": assistant}]}])
                flag = False
                continue
            conversation.extend(
                [{"role": "user", "content": [{"type": "text", "text": user}]}, {"role": "assistant", "content": [{"type": "text", "text": assistant}]}]
            )

        if len(history) == 0:
            conversation.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": message_text}]})
        else:
            conversation.append({"role": "user", "content": [{"type": "text", "text": message_text}]})
        print(f"prompt is -\n{conversation}")
        inputs = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_dict=True, tokenize=True, return_tensors="pt").to("cpu")
        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        image = Image.open(image)
        generation_kwargs = dict(
            pixel_values=torch.tensor(processor(image).pixel_values).to("cpu"),
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
        )
        generation_kwargs.update(inputs)

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

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
        description=f"Upload an image and start chatting about it, or simply try one of the examples below. If you won't upload an image, you will receive an error.",
        stop_btn="Stop Generation",
        multimodal=True,
    )

    return demo
