from pathlib import Path
import inspect
import requests
import gradio as gr
from PIL import Image
from threading import Thread
from transformers import TextIteratorStreamer

chat_template = """
{%- if messages[0][\"role\"] == \"system\" %}\n    {%- set system_message = messages[0][\"content\"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message[\"role\"] == \"user\" %}\n        {%- if loop.last and system_message is defined %}\n            {{- \"[INST]\" + system_message + \"\n\n\" }}\n        {%- else %}\n            {{- \"[INST]\" }}\n        {%- endif %}\n        {%- if message[\"content\"] is not string %}\n            {%- for chunk in message[\"content\"] %}\n                {%- if chunk[\"type\"] == \"text\" %}\n                    {{- chunk[\"content\"] }}\n                {%- elif chunk[\"type\"] == \"image\" %}\n                    {{- \"[IMG]\" }}\n                {%- else %}\n                    {{- raise_exception(\"Unrecognized content type!\") }}\n                {%- endif %}\n            {%- endfor %}\n        {%- else %}\n            {{- message[\"content\"] }}\n        {%- endif %}\n        {{- \"[/INST]\" }}\n    {%- elif message[\"role\"] == \"assistant\" %}\n        {{- message[\"content\"] + eos_token}}\n    {%- else %}\n        {{- raise_exception(\"Only user and assistant roles are supported, with the exception of an initial optional system message!\") }}\n    {%- endif %}\n{%- endfor %}
"""


def resize_with_aspect_ratio(image: Image, dst_height=512, dst_width=512):
    width, height = image.size
    if width > dst_width or height > dst_height:
        im_scale = min(dst_height / height, dst_width / width)
        resize_size = (int(width * im_scale), int(height * im_scale))
        return image.resize(resize_size)
    return image


def make_demo(model, processor):
    model_name = Path(model.config._name_or_path).parent.name

    example_image_urls = [
        ("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/dd5105d6-6a64-4935-8a34-3058a82c8d5d", "small.png"),
        ("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1221e2a8-a6da-413a-9af6-f04d56af3754", "chart.png"),
    ]

    for url, file_name in example_image_urls:
        if not Path(file_name).exists():
            Image.open(requests.get(url, stream=True).raw).save(file_name)
    if processor.chat_template is None:
        processor.set_chat_template(chat_template)

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
                raise gr.Error("You need to upload an image for Llama-3.2-Vision to work. Close the error and try again with an Image.")
        except NameError:
            # Handle the case where 'image' is not defined at all
            raise gr.Error("You need to upload an image for Llama-3.2-Vision to work. Close the error and try again with an Image.")

        conversation = []
        flag = False
        for user, assistant in history:
            if assistant is None:
                # pass
                flag = True
                conversation.extend([{"role": "user", "content": []}])
                continue
            if flag == True:
                conversation[0]["content"] = [{"type": "text", "content": f"{user}"}]
                conversation.append({"role": "assistant", "content": assistant})
                flag = False
                continue
            conversation.extend([{"role": "user", "content": [{"type": "text", "content": user}]}, {"role": "assistant", "content": assistant}])

        conversation.append({"role": "user", "content": [{"type": "text", "content": f"{message_text}"}, {"type": "image"}]})
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        print(f"prompt is -\n{prompt}")
        image = Image.open(image)
        image = resize_with_aspect_ratio(image)
        inputs = processor(prompt, image, return_tensors="pt")

        streamer = TextIteratorStreamer(
            processor,
            **{
                "skip_special_tokens": True,
                "skip_prompt": True,
                "clean_up_tokenization_spaces": False,
            },
        )
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            yield buffer

    has_additonal_buttons = "undo_button" in inspect.signature(gr.ChatInterface.__init__).parameters
    additional_buttons = {}
    if has_additonal_buttons:
        additional_buttons = {"undo_button": None, "retry_button": None}

    demo = gr.ChatInterface(
        fn=bot_streaming,
        title=f"{model_name} with OpenVINO",
        examples=[
            {"text": "What is the text saying?", "files": ["./small.png"]},
            {"text": "What does the chart display?", "files": ["./chart.png"]},
        ],
        description=f"{model_name} with OpenVINO. Upload an image and start chatting about it, or simply try one of the examples below. If you won't upload an image, you will receive an error.",
        stop_btn=None,
        multimodal=True,
        **additional_buttons,
    )

    return demo
