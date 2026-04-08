import gradio as gr
import torch
import os
import sys
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageOps
import fitz
import re
import warnings
import numpy as np
import base64
from io import StringIO, BytesIO
from pathlib import Path
import requests

MODEL_CONFIGS = {
    "Gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
    "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
}

TASK_PROMPTS = {
    "📋 Markdown": {"prompt": "<image>\n<|grounding|>Convert the document to markdown.", "has_grounding": True},
    "📝 Free OCR": {"prompt": "<image>\nFree OCR.", "has_grounding": False},
    "📍 Locate": {"prompt": "<image>\nLocate <|ref|>text<|/ref|> in the image.", "has_grounding": True},
    "🔍 Describe": {"prompt": "<image>\nDescribe this image in detail.", "has_grounding": False},
    "✏️ Custom": {"prompt": "", "has_grounding": False},
}

example_image_urls = [
    (
        "https://huggingface.co/spaces/merterbak/DeepSeek-OCR-Demo/resolve/main/examples/reachy-mini.jpg",
        "reachy-mini.jpg",
    ),
    (
        "https://huggingface.co/spaces/merterbak/DeepSeek-OCR-Demo/resolve/main/examples/ocr.jpg",
        "ocr.jpg",
    ),
]
for url, file_name in example_image_urls:
    if not Path(file_name).exists():
        img = Image.open(requests.get(url, stream=True, timeout=30).raw)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(file_name)


def make_demo(model, tokenizer):
    def extract_grounding_references(text):
        pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
        return re.findall(pattern, text, re.DOTALL)

    def draw_bounding_boxes(image, refs, extract_images=False):
        img_w, img_h = image.size
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
        draw2 = ImageDraw.Draw(overlay)
        font = ImageFont.load_default(size=30)
        crops = []

        color_map = {}
        np.random.seed(42)

        for ref in refs:
            label = ref[1]
            if label not in color_map:
                color_map[label] = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))

            color = color_map[label]
            coords = eval(ref[2])
            color_a = color + (60,)

            for box in coords:
                x1, y1, x2, y2 = int(box[0] / 999 * img_w), int(box[1] / 999 * img_h), int(box[2] / 999 * img_w), int(box[3] / 999 * img_h)

                if extract_images and label == "image":
                    crops.append(image.crop((x1, y1, x2, y2)))

                width = 5 if label == "title" else 3
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                draw2.rectangle([x1, y1, x2, y2], fill=color_a)

                text_bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                ty = max(0, y1 - 20)
                draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
                draw.text((x1 + 2, ty + 2), label, font=font, fill=(255, 255, 255))

        img_draw.paste(overlay, (0, 0), overlay)
        return img_draw, crops

    def clean_output(text, include_images=False, remove_labels=False):
        if not text:
            return ""
        pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
        matches = re.findall(pattern, text, re.DOTALL)
        img_num = 0

        for match in matches:
            if "<|ref|>image<|/ref|>" in match[0]:
                if include_images:
                    text = text.replace(match[0], f"\n\n**[Figure {img_num + 1}]**\n\n", 1)
                    img_num += 1
                else:
                    text = text.replace(match[0], "", 1)
            else:
                if remove_labels:
                    text = text.replace(match[0], "", 1)
                else:
                    text = text.replace(match[0], match[1], 1)

        return text.strip()

    def embed_images(markdown, crops):
        if not crops:
            return markdown
        for i, img in enumerate(crops):
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            markdown = markdown.replace(f"**[Figure {i + 1}]**", f"\n\n![Figure {i + 1}](data:image/png;base64,{b64})\n\n", 1)
        return markdown

    def process_image(image, mode, task, custom_prompt):
        if image is None:
            return " Error Upload image", "", "", None, []
        if task in ["✏️ Custom", "📍 Locate"] and not custom_prompt.strip():
            return "Enter prompt", "", "", None, []

        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        image = ImageOps.exif_transpose(image)

        config = MODEL_CONFIGS[mode]

        if task == "✏️ Custom":
            prompt = f"<image>\n{custom_prompt.strip()}"
            has_grounding = "<|grounding|>" in custom_prompt
        elif task == "📍 Locate":
            prompt = f"<image>\nLocate <|ref|>{custom_prompt.strip()}<|/ref|> in the image."
            has_grounding = True
        else:
            prompt = TASK_PROMPTS[task]["prompt"]
            has_grounding = TASK_PROMPTS[task]["has_grounding"]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(tmp.name, "JPEG", quality=95)
        tmp.close()
        out_dir = tempfile.mkdtemp()

        stdout = sys.stdout
        sys.stdout = StringIO()

        model.infer(
            tokenizer=tokenizer,
            prompt=prompt,
            image_file=tmp.name,
            output_path=out_dir,
            base_size=config["base_size"],
            image_size=config["image_size"],
            crop_mode=config["crop_mode"],
        )

        result = "\n".join(
            [l for l in sys.stdout.getvalue().split("\n") if not any(s in l for s in ["image:", "other:", "PATCHES", "====", "BASE:", "%|", "torch.Size"])]
        ).strip()
        sys.stdout = stdout

        os.unlink(tmp.name)
        shutil.rmtree(out_dir, ignore_errors=True)

        if not result:
            return "No text", "", "", None, []

        cleaned = clean_output(result, False, False)
        markdown = clean_output(result, True, True)

        img_out = None
        crops = []

        if has_grounding and "<|ref|>" in result:
            refs = extract_grounding_references(result)
            if refs:
                img_out, crops = draw_bounding_boxes(image, refs, True)

        markdown = embed_images(markdown, crops)

        return cleaned, markdown, result, img_out, crops

    def process_pdf(path, mode, task, custom_prompt, page_num):
        doc = fitz.open(path)
        total_pages = len(doc)
        if page_num < 1 or page_num > total_pages:
            doc.close()
            return f"Invalid page number. PDF has {total_pages} pages.", "", "", None, []
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
        img = Image.open(BytesIO(pix.tobytes("png")))
        doc.close()

        return process_image(img, mode, task, custom_prompt)

    def process_file(path, mode, task, custom_prompt, page_num):
        if not path:
            return "Error Upload file", "", "", None, []
        if path.lower().endswith(".pdf"):
            return process_pdf(path, mode, task, custom_prompt, page_num)
        else:
            return process_image(Image.open(path), mode, task, custom_prompt)

    def toggle_prompt(task):
        if task == "✏️ Custom":
            return gr.update(visible=True, label="Custom Prompt", placeholder="Add <|grounding|> for boxes")
        elif task == "📍 Locate":
            return gr.update(visible=True, label="Text to Locate", placeholder="Enter text")
        return gr.update(visible=False)

    def get_pdf_page_count(file_path):
        if not file_path or not file_path.lower().endswith(".pdf"):
            return 1
        doc = fitz.open(file_path)
        count = len(doc)
        doc.close()
        return count

    def load_image(file_path, page_num=1):
        if not file_path:
            return None
        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            page_idx = max(0, min(int(page_num) - 1, len(doc) - 1))
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
            img = Image.open(BytesIO(pix.tobytes("png")))
            doc.close()
            return img
        else:
            return Image.open(file_path)

    def update_page_selector(file_path):
        if not file_path:
            return gr.update(visible=False)
        if file_path.lower().endswith(".pdf"):
            page_count = get_pdf_page_count(file_path)
            return gr.update(visible=True, maximum=page_count, value=1, minimum=1, label=f"Select Page (1-{page_count})")
        return gr.update(visible=False)

    with gr.Blocks(theme=gr.themes.Soft(), title="DeepSeek-OCR") as demo:
        gr.Markdown("""
        # 🚀 DeepSeek-OCR Demo with OpenVINO
        **Convert documents to markdown, extract raw text, and locate specific content with bounding boxes.**
        
        **Hope this tool was helpful! If so, a quick like ❤️ would mean a lot :)**
        """)

        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(label="Upload Image or PDF", file_types=["image", ".pdf"], type="filepath")
                input_img = gr.Image(label="Input Image", type="pil", height=300)
                page_selector = gr.Number(label="Select Page", value=1, minimum=1, step=1, visible=False)
                mode = gr.Dropdown(list(MODEL_CONFIGS.keys()), value="Gundam", label="Mode")
                task = gr.Dropdown(list(TASK_PROMPTS.keys()), value="📋 Markdown", label="Task")
                prompt = gr.Textbox(label="Prompt", lines=2, visible=False)
                btn = gr.Button("Extract", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📝 Text"):
                        text_out = gr.Textbox(lines=20, show_copy_button=True, show_label=False)
                    with gr.Tab("🎨 Markdown"):
                        md_out = gr.Markdown("")
                    with gr.Tab("🖼️ Boxes"):
                        img_out = gr.Image(type="pil", height=500, show_label=False)
                    with gr.Tab("🖼️ Cropped Images"):
                        gallery = gr.Gallery(show_label=False, columns=3, height=400)
                    with gr.Tab("🔍 Raw"):
                        raw_out = gr.Textbox(lines=20, show_copy_button=True, show_label=False)

        gr.Examples(
            examples=[["ocr.jpg", "Gundam", "📋 Markdown", ""], ["reachy-mini.jpg", "Gundam", "📍 Locate", "Robot"]],
            inputs=[input_img, mode, task, prompt],
            cache_examples=False,
        )

        with gr.Accordion("ℹ️ Info", open=False):
            gr.Markdown("""
            ### Modes
            - **Gundam**: 1024 base + 640 tiles with cropping - Best balance
            - **Tiny**: 512×512, no crop - Fastest
            - **Small**: 640×640, no crop - Quick
            - **Base**: 1024×1024, no crop - Standard
            - **Large**: 1280×1280, no crop - Highest quality
            
            ### Tasks
            - **Markdown**: Convert document to structured markdown (grounding ✅)
            - **Free OCR**: Simple text extraction
            - **Locate**: Find specific things in image (grounding ✅)
            - **Describe**: General image description
            - **Custom**: Your own prompt (add `<|grounding|>` for boxes)
            """)

        file_in.change(load_image, [file_in, page_selector], [input_img])
        file_in.change(update_page_selector, [file_in], [page_selector])
        page_selector.change(load_image, [file_in, page_selector], [input_img])
        task.change(toggle_prompt, [task], [prompt])

        def run(image, file_path, mode, task, custom_prompt, page_num):
            if file_path:
                return process_file(file_path, mode, task, custom_prompt, int(page_num))
            if image is not None:
                return process_image(image, mode, task, custom_prompt)
            return "Error uploading file or image", "", "", None, []

        btn.click(run, [input_img, file_in, mode, task, prompt, page_selector], [text_out, md_out, raw_out, img_out, gallery])
        return demo
