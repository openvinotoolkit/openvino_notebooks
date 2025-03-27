from pathlib import Path
import requests
import torch
from PIL import Image
import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread


MODEL_ID = "THUDM/glm-4v-9b"
MODEL_NAME = MODEL_ID.split("/")[-1]

TITLE = f'<h1>OpenVINO Visual Language Chat</h1><br><center>🚀 MODEL NOW: <a href="https://hf.co/{MODEL_ID}">{MODEL_NAME}</a></center>'

DESCRIPTION = f"""
<center>
<p>
<br>
✨ Tips: Send Messages or upload 1 IMAGE per time.
<br>
✨ Tips: Please increase MAX LENGTH when deal with file.
<br>
✨ Supported Format: png, jpg, webp, bmp, tiff
</p>
</center>"""

CSS = """
h1 {
    text-align: center;
    display: block;
}
"""


def download_example_images():
    example_images = {
        "cat.png": "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11",
        "hotel.jpg": "https://github.com/user-attachments/assets/1dceb84c-5e71-499c-a85c-89a433c59217",
        "spacecat.png": "https://github.com/user-attachments/assets/a42d10a7-9fd6-478d-8929-45b770c5f6d4",
    }

    for path, url in example_images.items():
        if not Path(path).exists():
            image = Image.open(requests.get(url, stream=True).raw)
            image.save(path)


def make_demo(model, tokenizer):
    def mode_load(path):
        choice = ""
        file_type = path.split(".")[-1]
        print(file_type)
        if file_type in ["png", "jpg", "jpeg", "bmp", "tiff", "webp"]:
            content = Image.open(path).convert("RGB")
            choice = "image"
            return choice, content
        else:
            raise gr.Error("Oops, unsupported files.")

    def stream_chat(message, history: list, temperature: float, max_length: int, top_p: float, top_k: int, penalty: float):
        print(f"message is - {message}")
        print(f"history is - {history}")
        conversation = []
        prompt_files = []
        if message["files"]:
            choice, contents = mode_load(message["files"][-1])
            if choice == "image":
                conversation.append({"role": "user", "image": contents, "content": message["text"]})
        else:
            if len(history) == 0:
                # raise gr.Error("Please upload an image first.")
                contents = None
                conversation.append({"role": "user", "content": message["text"]})
            else:
                # image = Image.open(history[0][0][0])
                for prompt, answer in history:
                    if answer is None:
                        prompt_files.append(prompt[0])
                        conversation.extend([{"role": "user", "content": ""}, {"role": "assistant", "content": ""}])
                    else:
                        conversation.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])
                if prompt_files:
                    choice, contents = mode_load(prompt_files[-1])
                    if choice == "image":
                        conversation.append({"role": "user", "image": contents, "content": message["text"]})
                    elif choice == "doc":
                        format_msg = contents + "\n\n\n" + "{} files uploaded.\n" + message["text"]
                        conversation.append({"role": "user", "content": format_msg})
        print(f"Conversation is -\n{conversation}")

        input_ids = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(
            model.device
        )
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            max_length=max_length,
            streamer=streamer,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=penalty,
            eos_token_id=[151329, 151336, 151338],
        )
        gen_kwargs = {**input_ids, **generate_kwargs}

        with torch.no_grad():
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()
            buffer = ""
            for new_text in streamer:
                buffer += new_text
                yield buffer

    chatbot = gr.Chatbot(label="Chatbox", height=600, placeholder=DESCRIPTION)
    chat_input = gr.MultimodalTextbox(
        interactive=True,
        placeholder="Enter message or upload a file one time...",
        show_label=False,
    )
    download_example_images()
    EXAMPLES = [
        [{"text": "Describe this image in great detailed.", "files": ["./cat.png"]}],
        [{"text": "Please describe this image and guess where it is?", "files": ["./hotel.jpg"]}],
        [{"text": "What's in the image, is it real happen?", "files": ["./spacecat.png"]}],
    ]

    with gr.Blocks(css=CSS, theme="soft", fill_height=True) as demo:
        gr.HTML(TITLE)
        gr.ChatInterface(
            fn=stream_chat,
            multimodal=True,
            textbox=chat_input,
            chatbot=chatbot,
            fill_height=True,
            additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
            additional_inputs=[
                gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=0.8,
                    label="Temperature",
                    render=False,
                ),
                gr.Slider(
                    minimum=1024,
                    maximum=8192,
                    step=1,
                    value=4096,
                    label="Max Length",
                    render=False,
                ),
                gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=1.0,
                    label="top_p",
                    render=False,
                ),
                gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=10,
                    label="top_k",
                    render=False,
                ),
                gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    label="Repetition penalty",
                    render=False,
                ),
            ],
        ),
        gr.Examples(EXAMPLES, [chat_input])

    return demo
