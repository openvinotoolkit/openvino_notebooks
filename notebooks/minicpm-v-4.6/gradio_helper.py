import threading
from pathlib import Path

import gradio as gr
from PIL import Image


def make_demo(ov_model, processor):
    def bot_streaming(message, history):
        # Extract image from the message
        image = None
        if message.get("files"):
            image = Image.open(message["files"][-1]).convert("RGB")

        text = message.get("text", "")
        if not text:
            yield "Please provide a text message."
            return

        # Build messages in the same format as the model card
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]

        downsample_mode = "16x"

        # Use processor.apply_chat_template directly (aligned with model card)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            downsample_mode=downsample_mode,
            max_slice_nums=36,
        )

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_complete = threading.Event()

        def generate_and_signal_complete():
            generate_kwargs = dict(
                **inputs,
                downsample_mode=downsample_mode,
                max_new_tokens=512,
                do_sample=False,
                streamer=streamer,
            )
            ov_model.generate(**generate_kwargs)
            generation_complete.set()

        t = threading.Thread(target=generate_and_signal_complete)
        t.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            yield buffer

    demo = gr.ChatInterface(
        fn=bot_streaming,
        title="MiniCPM-V 4.6 with OpenVINO",
        description="Upload an image and ask questions about it.",
        multimodal=True,
        textbox=gr.MultimodalTextbox(placeholder="Type a message or upload an image...", scale=7),
    )

    return demo
