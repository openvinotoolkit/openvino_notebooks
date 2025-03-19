import base64
import os
import shutil
import time
import uuid
from pathlib import Path
import requests

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from transformers.image_utils import load_image

title = """# GOT-OCR 2.0: OpenVINO implementation demo"""

description = """
This demo utilizes the **GOT-OCR 2.0** to extract text from images.
Explore the capabilities of this cutting-edge model through this interactive demo!
"""

tasks = [
    "Plain Text OCR",
    "Format Text OCR",
    "Fine-grained OCR (Box)",
    "Fine-grained OCR (Color)",
    "Multi-crop OCR",
    "Multi-page OCR",
]

ocr_types = ["ocr", "format"]
ocr_colors = ["red", "green", "blue"]

punctuation_dict = {
    "，": ",",
    "。": ".",
}
translation_table = str.maketrans(punctuation_dict)
stop_str = "<|im_end|>"


def svg_to_html(svg_content, output_filename):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SVG Embedded in HTML</title>
    </head>
    <body>
        <svg width="2100" height="15000" xmlns="http://www.w3.org/2000/svg">
            {svg_content}
        </svg>
    </body>
    </html>
    """

    with open(output_filename, "w") as file:
        file.write(html_content)


def render_ocr_text(text, result_path, format_text=False):
    if text.endswith(stop_str):
        text = text[: -len(stop_str)]
    text = text.strip()

    if "**kern" in text:
        import verovio

        tk = verovio.toolkit()
        tk.loadData(text)
        tk.setOptions(
            {
                "pageWidth": 2100,
                "pageHeight": 800,
                "footer": "none",
                "barLineWidth": 0.5,
                "beamMaxSlope": 15,
                "staffLineWidth": 0.2,
                "spacingStaff": 6,
            }
        )
        tk.getPageCount()
        svg = tk.renderToSVG()
        svg = svg.replace('overflow="inherit"', 'overflow="visible"')

        svg_to_html(svg, result_path)

    if format_text and "**kern" not in text:
        if "\\begin{tikzpicture}" not in text:
            html_path = "content-mmd-to-html.html"
            right_num = text.count("\\right")
            left_num = text.count("\left")

            if right_num != left_num:
                text = (
                    text.replace("\left(", "(")
                    .replace("\\right)", ")")
                    .replace("\left[", "[")
                    .replace("\\right]", "]")
                    .replace("\left{", "{")
                    .replace("\\right}", "}")
                    .replace("\left|", "|")
                    .replace("\\right|", "|")
                    .replace("\left.", ".")
                    .replace("\\right.", ".")
                )

            text = text.replace('"', "``").replace("$", "")

            outputs_list = text.split("\n")
            gt = ""
            for out in outputs_list:
                gt += '"' + out.replace("\\", "\\\\") + r"\n" + '"' + "+" + "\n"

            gt = gt[:-2]

            with open(html_path, "r") as web_f:
                lines = web_f.read()
                lines = lines.split("const text =")
                new_web = lines[0] + "const text =" + gt + lines[1]
        else:
            html_path = "tikz.html"
            text = text.translate(translation_table)
            outputs_list = text.split("\n")
            gt = ""
            for out in outputs_list:
                if out:
                    if "\\begin{tikzpicture}" not in out and "\\end{tikzpicture}" not in out:
                        while out[-1] == " ":
                            out = out[:-1]
                            if out is None:
                                break

                        if out:
                            if out[-1] != ";":
                                gt += out[:-1] + ";\n"
                            else:
                                gt += out + "\n"
                    else:
                        gt += out + "\n"

            with open(html_path, "r") as web_f:
                lines = web_f.read()
                lines = lines.split("const text =")
                new_web = lines[0] + gt + lines[1]

        with open(result_path, "w") as web_f_new:
            web_f_new.write(new_web)


def download_examples():
    example_images = {
        "sheet_music.png": "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/sheet_music.png",
        "latex.png": "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/latex.png",
        "multi_box.png": "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png",
    }

    for file_name, url in example_images.items():
        if not Path(file_name).exists():
            Image.open(requests.get(url, stream=True).raw).save(file_name)


def download_rendering_tools():
    if not Path("content-mmd-to-html.html").exists():
        r = requests.get("https://huggingface.co/spaces/yonigozlan/GOT-OCR-Transformers/raw/main/render_tools/content-mmd-to-html.html")
        with open("content-mmd-to-html.html", "w") as f:
            f.write(r.text)

    if not Path("tikz.html").exists():
        r = requests.get("https://huggingface.co/spaces/yonigozlan/GOT-OCR-Transformers/raw/main/render_tools/tikz.html")
        with open("tikz.html", "w") as f:
            f.write(r.text)


def make_demo(model, processor):
    UPLOAD_FOLDER = "./uploads"
    RESULTS_FOLDER = "./results"
    stop_str = "<|im_end|>"
    for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    def cleanup_old_files():
        current_time = time.time()
        for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
            for file_path in Path(folder).glob("*"):
                if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
                    file_path.unlink()

    cleanup_old_files()
    download_rendering_tools()
    download_examples()

    def process_image(image, task, ocr_type=None, ocr_box=None, ocr_color=None):
        if image is None:
            return "Error: No image provided", None, None

        unique_id = str(uuid.uuid4())
        image_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.png")
        result_path = os.path.join(RESULTS_FOLDER, f"{unique_id}.html")
        try:
            if not isinstance(image, (tuple, list)):
                image = [image]
            else:
                image = [img[0] for img in image]
            for i, img in enumerate(image):
                if isinstance(img, dict):
                    composite_image = img.get("composite")
                    if composite_image is not None:
                        if isinstance(composite_image, np.ndarray):
                            cv2.imwrite(image_path, cv2.cvtColor(composite_image, cv2.COLOR_RGB2BGR))
                        elif isinstance(composite_image, Image.Image):
                            composite_image.save(image_path)
                        else:
                            return (
                                "Error: Unsupported image format from ImageEditor",
                                None,
                                None,
                            )
                    else:
                        return (
                            "Error: No composite image found in ImageEditor output",
                            None,
                            None,
                        )
                elif isinstance(img, np.ndarray):
                    cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                elif isinstance(img, str):
                    shutil.copy(img, image_path)
                else:
                    return "Error: Unsupported image format", None, None

                image[i] = load_image(image_path)

            if task == "Plain Text OCR":
                inputs = processor(image, return_tensors="pt")
                generate_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=processor.tokenizer,
                    stop_strings=stop_str,
                    max_new_tokens=4096,
                )
                res = processor.decode(
                    generate_ids[0, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                return res, None, unique_id
            else:
                if task == "Format Text OCR":
                    inputs = processor(image, return_tensors="pt", format=True)
                    generate_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=processor.tokenizer,
                        stop_strings=stop_str,
                        max_new_tokens=4096,
                    )
                    res = processor.decode(
                        generate_ids[0, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                    ocr_type = "format"
                elif task == "Fine-grained OCR (Box)":
                    inputs = processor(image, return_tensors="pt", box=ocr_box)
                    generate_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=processor.tokenizer,
                        stop_strings=stop_str,
                        max_new_tokens=4096,
                    )
                    res = processor.decode(
                        generate_ids[0, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                elif task == "Fine-grained OCR (Color)":
                    inputs = processor(image, return_tensors="pt", color=ocr_color)
                    generate_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=processor.tokenizer,
                        stop_strings=stop_str,
                        max_new_tokens=4096,
                    )
                    res = processor.decode(
                        generate_ids[0, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                elif task == "Multi-crop OCR":
                    inputs = processor(
                        image,
                        return_tensors="pt",
                        format=True,
                        crop_to_patches=True,
                        max_patches=5,
                    )
                    generate_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=processor.tokenizer,
                        stop_strings=stop_str,
                        max_new_tokens=4096,
                    )
                    res = processor.decode(
                        generate_ids[0, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                    ocr_type = "format"
                elif task == "Multi-page OCR":
                    inputs = processor(image, return_tensors="pt", multi_page=True, format=True)
                    generate_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=processor.tokenizer,
                        stop_strings=stop_str,
                        max_new_tokens=4096,
                    )
                    res = processor.decode(
                        generate_ids[0, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                    ocr_type = "format"

                render_ocr_text(res, result_path, format_text=ocr_type == "format")
                if os.path.exists(result_path):
                    with open(result_path, "r") as f:
                        html_content = f.read()
                    return res, html_content, unique_id
                else:
                    return res, None, unique_id
        except Exception as e:
            return f"Error: {str(e)}", None, None
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

    def update_image_input(task):
        if task == "Fine-grained OCR (Color)":
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        elif task == "Multi-page OCR":
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        else:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    def update_inputs(task):
        if task in [
            "Plain Text OCR",
            "Format Text OCR",
            "Multi-crop OCR",
        ]:
            return [
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ]
        elif task == "Fine-grained OCR (Box)":
            return [
                gr.update(visible=True, choices=["ocr", "format"]),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ]
        elif task == "Fine-grained OCR (Color)":
            return [
                gr.update(visible=True, choices=["ocr", "format"]),
                gr.update(visible=False),
                gr.update(visible=True, choices=["red", "green", "blue"]),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            ]
        elif task == "Multi-page OCR":
            return [
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
            ]

    def ocr_demo(image, task, ocr_type, ocr_box, ocr_color):
        res, html_content, unique_id = process_image(image, task, ocr_type, ocr_box, ocr_color)

        if isinstance(res, str) and res.startswith("Error:"):
            return res, None

        res = res.replace("\\title", "\\title ")
        formatted_res = res
        # formatted_res = parse_latex_output(res)

        if html_content:
            encoded_html = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
            iframe_src = f"data:text/html;base64,{encoded_html}"
            iframe = f'<iframe src="{iframe_src}" width="100%" height="600px"></iframe>'
            download_link = f'<a href="data:text/html;base64,{encoded_html}" download="result_{unique_id}.html">Download Full Result</a>'
            return formatted_res, f"{download_link}<br>{iframe}"
        return formatted_res, None

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    image_input = gr.Image(type="filepath", label="Input Image")
                    gallery_input = gr.Gallery(type="filepath", label="Input images", visible=False)
                    image_editor = gr.ImageEditor(label="Image Editor", type="pil", visible=False)
                    task_dropdown = gr.Dropdown(
                        choices=[
                            "Plain Text OCR",
                            "Format Text OCR",
                            "Fine-grained OCR (Box)",
                            "Fine-grained OCR (Color)",
                            "Multi-crop OCR",
                            "Multi-page OCR",
                        ],
                        label="Select Task",
                        value="Plain Text OCR",
                    )
                    ocr_type_dropdown = gr.Dropdown(choices=["ocr", "format"], label="OCR Type", visible=False)
                    ocr_box_input = gr.Textbox(
                        label="OCR Box (x1,y1,x2,y2)",
                        placeholder="[100,100,200,200]",
                        visible=False,
                    )
                    ocr_color_dropdown = gr.Dropdown(choices=["red", "green", "blue"], label="OCR Color", visible=False)
                    # with gr.Row():
                    # max_new_tokens_slider = gr.Slider(50, 500, step=10, value=150, label="Max New Tokens")
                    # no_repeat_ngram_size_slider = gr.Slider(1, 10, step=1, value=2, label="No Repeat N-gram Size")

                    submit_button = gr.Button("Process", variant="primary")
                    editor_submit_button = gr.Button("Process Edited Image", visible=False, variant="primary")
                    gallery_submit_button = gr.Button("Process Multiple Images", visible=False, variant="primary")

            with gr.Column(scale=1):
                with gr.Group():
                    output_markdown = gr.Textbox(label="Text output")
                    output_html = gr.HTML(label="HTML output")

        input_types = [
            image_input,
            image_editor,
            gallery_input,
        ]

        task_dropdown.change(
            update_inputs,
            inputs=[task_dropdown],
            outputs=[
                ocr_type_dropdown,
                ocr_box_input,
                ocr_color_dropdown,
                image_input,
                image_editor,
                submit_button,
                editor_submit_button,
                gallery_input,
                gallery_submit_button,
            ],
        )

        task_dropdown.change(
            update_image_input,
            inputs=[task_dropdown],
            outputs=[
                image_input,
                image_editor,
                editor_submit_button,
                gallery_input,
                gallery_submit_button,
            ],
        )

        submit_button.click(
            ocr_demo,
            inputs=[
                image_input,
                task_dropdown,
                ocr_type_dropdown,
                ocr_box_input,
                ocr_color_dropdown,
            ],
            outputs=[output_markdown, output_html],
        )
        editor_submit_button.click(
            ocr_demo,
            inputs=[
                image_editor,
                task_dropdown,
                ocr_type_dropdown,
                ocr_box_input,
                ocr_color_dropdown,
            ],
            outputs=[output_markdown, output_html],
        )
        gallery_submit_button.click(
            ocr_demo,
            inputs=[
                gallery_input,
                task_dropdown,
                ocr_type_dropdown,
                ocr_box_input,
                ocr_color_dropdown,
            ],
            outputs=[output_markdown, output_html],
        )
        example = gr.Examples(
            examples=[
                [
                    "./sheet_music.png",
                    "Format Text OCR",
                    "format",
                    None,
                    None,
                ],
                [
                    "./latex.png",
                    "Format Text OCR",
                    "format",
                    None,
                    None,
                ],
            ],
            inputs=[
                image_input,
                task_dropdown,
                ocr_type_dropdown,
                ocr_box_input,
                ocr_color_dropdown,
            ],
            outputs=[output_markdown, output_html],
        )
        example_finegrained = gr.Examples(
            examples=[
                [
                    "./multi_box.png",
                    "Fine-grained OCR (Color)",
                    "ocr",
                    None,
                    "red",
                ]
            ],
            inputs=[
                image_editor,
                task_dropdown,
                ocr_type_dropdown,
                ocr_box_input,
                ocr_color_dropdown,
            ],
            outputs=[output_markdown, output_html],
            label="Fine-grained example",
        )

    return demo
