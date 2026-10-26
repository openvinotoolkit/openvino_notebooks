from threading import Thread

import gradio as gr
from transformers import TextIteratorStreamer

PROMPTS = {
    "Transcribe text": "Transcribe all the text in this image.",
    "Convert to Markdown": "Convert this page to markdown.",
    "Ask a question": "",
}


def make_demo(ov_model, processor, examples=()):
    downsample_mode = ov_model.config.downsample_mode

    def run_ocr(image, prompt, enable_thinking, max_new_tokens):
        if image is None or not prompt.strip():
            raise gr.Error("Upload an image and enter a prompt.")
        messages = [{"role": "user", "content": [{"type": "image", "image": image.convert("RGB")}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
            processor_kwargs={"downsample_mode": downsample_mode},
        )
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation = Thread(
            target=ov_model.generate,
            kwargs=dict(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, streamer=streamer),
        )
        generation.start()
        text = ""
        for chunk in streamer:
            text += chunk
            yield text
        generation.join()

    with gr.Blocks(title="MiniCPM-OCR with OpenVINO") as demo:
        gr.Markdown("# MiniCPM-OCR with OpenVINO")
        gr.Markdown(f"Upload a document, screenshot or photo with text and choose a task ({downsample_mode} visual downsampling).")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image")
                task = gr.Radio(list(PROMPTS), value="Transcribe text", label="Task")
                prompt = gr.Textbox(value=PROMPTS["Transcribe text"], label="Prompt")
                with gr.Accordion("Generation settings", open=False):
                    enable_thinking = gr.Checkbox(value=False, label="Enable thinking")
                    max_new_tokens = gr.Slider(64, 2048, value=1024, step=64, label="Max new tokens")
                submit = gr.Button("Run OCR", variant="primary")
            output = gr.Textbox(label="Result", lines=24)
        if examples:
            gr.Examples(examples=[[path] for path in examples], inputs=[image])
        task.change(lambda name: PROMPTS[name], inputs=task, outputs=prompt)
        submit.click(run_ocr, inputs=[image, prompt, enable_thinking, max_new_tokens], outputs=output)
    return demo
