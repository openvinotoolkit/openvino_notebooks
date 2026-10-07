from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from queue import Empty

import gradio as gr
import openvino as ov
import requests
from PIL import Image, ImageDraw, ImageOps


@lru_cache(maxsize=1)
def load_model(model_dir: str, device: str):
    from optimum.intel import OVModelForVisualCausalLM
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir)
    model = OVModelForVisualCausalLM.from_pretrained(model_dir, device=device)
    return processor, model


def stream_generate(processor, model, inputs, *, max_new_tokens):
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.5)
    with ThreadPoolExecutor(max_workers=1) as executor:
        generation = executor.submit(
            model.generate, **inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer
        )
        while True:
            try:
                chunk = next(streamer)
            except Empty:
                if generation.done():
                    generation.result()
                    raise RuntimeError("Generation finished without closing the text stream.")
                continue
            except StopIteration:
                break
            if chunk:
                yield chunk
        generation.result()


def _example_images(export_root: Path):
    example_dir = export_root / "demo_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    sample_paths = {name: example_dir / f"{name}.png" for name in ("cat", "bee", "invoice")}
    image_urls = {
        "cat": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "bee": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    }
    for name, url in image_urls.items():
        if not sample_paths[name].is_file():
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with Image.open(BytesIO(response.content)) as image:
                image.convert("RGB").save(sample_paths[name])

    if not sample_paths["invoice"].is_file():
        invoice = Image.new("RGB", (420, 220), "white")
        draw = ImageDraw.Draw(invoice)
        draw.text((20, 30), "INVOICE # OV-4721", fill="black")
        draw.text((20, 80), "DUE DATE: 2026-11-15", fill="black")
        draw.text((20, 130), "TOTAL: 125 USD", fill="black")
        invoice.resize((1680, 880), Image.Resampling.NEAREST).save(sample_paths["invoice"])

    preview_path = example_dir / "cat-and-bee.png"
    if not preview_path.is_file():
        preview = Image.new("RGB", (640, 320), "white")
        for index, name in enumerate(("cat", "bee")):
            with Image.open(sample_paths[name]) as image:
                preview.paste(ImageOps.pad(image, (320, 320), color="white"), (index * 320, 0))
        preview.save(preview_path)

    return sample_paths, preview_path


def make_demo(export_root=Path("MiniCPM-V-4.7-ov")):
    export_root = Path(export_root).resolve()
    model_dirs = sorted({path.parent for path in export_root.rglob("openvino_language_model.xml")})
    if not model_dirs:
        raise FileNotFoundError(f"No exported MiniCPM-V 4.7 models found in {export_root}. Run the export cell first.")

    devices = [device for device in ov.Core().available_devices if device != "NPU"]
    if not devices:
        raise RuntimeError("No supported OpenVINO device is available for the Gradio demo.")

    sample_paths, preview_path = _example_images(export_root)
    examples = [
        ([str(sample_paths["cat"])], "Describe this image.", False, str(sample_paths["cat"])),
        (
            [str(sample_paths["cat"]), str(sample_paths["bee"])],
            "What is shown in the first and second images?",
            False,
            str(preview_path),
        ),
        (
            [str(sample_paths["invoice"])],
            "What are the invoice number and due date?",
            False,
            str(sample_paths["invoice"]),
        ),
        ([str(sample_paths["cat"])], "Describe this image in detail.", True, str(sample_paths["cat"])),
    ]

    def ask_images(image_files, question, enable_thinking, selected_model, selected_device):
        if not image_files or not question.strip():
            raise ValueError("Upload at least one image and enter a question.")
        model_path = Path(selected_model).resolve()
        if model_path not in model_dirs or selected_device not in devices:
            raise ValueError("Select an exported model and an available device.")

        processor, model = load_model(str(model_path), selected_device)
        content = []
        for image_file in image_files:
            with Image.open(image_file) as image:
                content.append({"type": "image", "image": image.convert("RGB")})
        content.append({"type": "text", "text": question})
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
            processor_kwargs={"use_image_id": True},
        )
        response = "<think>" if enable_thinking else ""
        for chunk in stream_generate(processor, model, inputs, max_new_tokens=1024 if enable_thinking else 128):
            response += chunk
            yield [{"role": "assistant", "content": response}]

    def load_example(evt: gr.SelectData):
        return examples[evt.index][:3]

    with gr.Blocks() as demo:
        gr.Markdown("# MiniCPM-V 4.7 with OpenVINO")
        gr.Markdown("Ask a question about one or more images (single request, 16x downsampling).")
        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(
                    choices=[(str(path.relative_to(export_root)), str(path)) for path in model_dirs],
                    value=str(model_dirs[0]),
                    label="Exported model",
                )
                device_choice = gr.Dropdown(
                    choices=devices, value="CPU" if "CPU" in devices else devices[0], label="Device"
                )
                image_files = gr.File(label="Images", file_count="multiple", file_types=["image"], type="filepath")
                question = gr.Textbox(label="Question")
                enable_thinking = gr.Checkbox(label="Enable thinking", value=False)
                submit = gr.Button("Submit", variant="primary")
            answer = gr.Chatbot(label="Answer", height=400, reasoning_tags=[("<think>", "</think>")])
        previews = gr.Dataset(
            components=[
                gr.Image(label="Preview", type="filepath", render=False),
                gr.Textbox(label="Question", render=False),
                gr.Checkbox(label="Enable thinking", render=False),
            ],
            samples=[[preview, question, thinking] for _, question, thinking, preview in examples],
            label="Examples (click a row to load)",
            layout="table",
            type="index",
        )
        previews.select(load_example, outputs=[image_files, question, enable_thinking])
        submit.click(
            ask_images,
            inputs=[image_files, question, enable_thinking, model_choice, device_choice],
            outputs=answer,
        )
    return demo
