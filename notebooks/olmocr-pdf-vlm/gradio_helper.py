import base64
import gradio as gr
from io import BytesIO
from pathlib import Path
from PIL import Image
import json
import requests

import numpy as np
import openvino as ov

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text


sample_path = Path("./paper.pdf")

if not sample_path.exists():
    r = requests.get("https://olmocr.allenai.org/papers/olmocr.pdf")
    with sample_path.open("wb") as f:
        f.write(r.content)


def process_file_upload(
    file,
    page_number=1,
):
    image_base64 = render_pdf_to_base64png(file, page_number, target_longest_image_dim=1024)

    anchor_text = get_anchor_text(file, page_number, pdf_engine="pdfreport", target_length=4000)
    prompt = build_finetuning_prompt(anchor_text)

    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    return prompt, main_image


def make_demo(pipe):
    def generate(file_input, page_number, temperature, max_new_tokens):
        prompt, image = process_file_upload(file_input, page_number=page_number)
        image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
        output = pipe.generate(
            prompt=prompt,
            image=ov.Tensor(image_data),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            stop_strings="<|im_end|>",
            do_sample=True,
        )
        try:
            result = json.loads(output.texts[0])["natural_text"]
        except json.decoder.JSONDecodeError as exc:
            result = output.texts[0]

        return result, image

    with gr.Blocks(title="olmOCR Document Analyzer") as demo:
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="Upload a PDF file",
                    file_types=[".pdf"],
                )

                with gr.Row():
                    with gr.Column():
                        page_number = gr.Number(label="Page Number", value=1, minimum=1, step=1)
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.8,
                            step=0.1,
                        )
                        max_new_tokens = gr.Slider(
                            label="Max New Tokens",
                            minimum=10,
                            maximum=5000,
                            value=4096,
                            step=10,
                        )

                submit_btn_file = gr.Button("Analyze File", variant="primary")
                example = gr.Examples(
                    examples=[
                        [
                            "./paper.pdf",
                            1,
                            0.8,
                            4096,
                            None,
                        ],
                    ],
                    inputs=[
                        file_input,
                        page_number,
                        temperature,
                        max_new_tokens,
                    ],
                )

            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column():
                        image_output = gr.Image(label="Image/PDF Page", type="pil")

                    with gr.Column():
                        text_output = gr.Textbox(label="Result", lines=10)

            submit_btn_file.click(
                fn=generate,
                inputs=[
                    file_input,
                    page_number,
                    temperature,
                    max_new_tokens,
                ],
                outputs=[text_output, image_output],
            )

    return demo
