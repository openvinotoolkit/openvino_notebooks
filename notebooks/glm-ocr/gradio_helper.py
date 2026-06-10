from __future__ import annotations

import tempfile
from pathlib import Path
from threading import Thread

import gradio as gr
from PIL import Image, ImageOps
from transformers import TextIteratorStreamer

TASK_PROMPTS = {
    "Text": "Text Recognition:",
    "Formula": "Formula Recognition:",
    "Table": "Table Recognition:",
}


# Sample images are hosted as GitHub user-attachments; downloaded on first run
# via :func:`download_example_assets` (the notebook calls it from its
# prerequisites cell). Keeping these out of the repository avoids shipping
# binary blobs in the PR.
EXAMPLE_ASSET_URLS: dict[str, str] = {
    "ocr_text_sample.png": "https://github.com/user-attachments/assets/fda2c6f6-bbba-4ece-b90b-c42bca60e525",
    "ocr_formula_sample.png": "https://github.com/user-attachments/assets/c9407e8e-3f2a-4d74-807b-dcefa06a3741",
    "ocr_table_sample.png": "https://github.com/user-attachments/assets/4c64fbf7-5181-4ccc-bb0a-92ea43b3dcc8",
    "ocr_doc_sample.png": "https://github.com/user-attachments/assets/5b482249-ad47-4c32-8c89-302a60b3c71e",
}


example_files = [
    ("ocr_text_sample.png", "Text"),
    ("ocr_formula_sample.png", "Formula"),
    ("ocr_table_sample.png", "Table"),
]


def _build_inputs(processor, image: Image.Image, task: str):
    image = ImageOps.exif_transpose(image.convert("RGB"))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name, format="PNG")
    tmp.close()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": tmp.name},
                {"type": "text", "text": TASK_PROMPTS.get(task, TASK_PROMPTS["Text"])},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs


def _result_textbox(label: str, lines: int = 24):
    import inspect as _inspect

    kw = {"label": label, "lines": lines}
    params = _inspect.signature(gr.Textbox).parameters
    if "show_copy_button" in params:
        kw["show_copy_button"] = True
    if "autoscroll" in params:
        kw["autoscroll"] = True
    return gr.Textbox(**kw)


def make_demo(model, processor, detector=None):
    """Build a Gradio demo for GLM-OCR.

    If ``detector`` is provided (a :class:`pp_doclayout_v3_helper.LayoutDetector`),
    a second tab runs the full document-parsing pipeline
    (PP-DocLayoutV3 + GLM-OCR).
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    def run_single(image, task, max_new_tokens):
        if image is None:
            yield "[ERROR] Please upload an image first."
            return
        inputs = _build_inputs(processor, image, task)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        gen_kwargs = dict(
            inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        buf = ""
        for piece in streamer:
            buf += piece
            yield buf
        thread.join(timeout=1.0)

    def run_pipeline(image, max_new_tokens, score_thr):
        if image is None:
            return None, "[ERROR] Please upload a document image first."
        if detector is None:
            return None, "[ERROR] Pipeline is not available — no detector was provided."
        from pp_doclayout_v3_helper import draw_layout, run_pipeline as _run_pipeline

        image = ImageOps.exif_transpose(image.convert("RGB"))
        detections = detector(image, score_thr=float(score_thr))
        annotated = draw_layout(image, detections)
        md = _run_pipeline(
            detector,
            model,
            processor,
            image,
            max_new_tokens=int(max_new_tokens),
            score_thr=float(score_thr),
        )
        summary = f"Detected {len(detections)} regions.\n\n{md}"
        return annotated, summary

    with gr.Blocks(title="GLM-OCR + OpenVINO") as demo:
        gr.Markdown("# GLM-OCR + OpenVINO\n" "Upload a document image and run GLM-OCR through optimum-intel + OpenVINO.")
        with gr.Tab("Single-region recognition"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_in = gr.Image(type="pil", label="Document image")
                    task = gr.Radio(
                        choices=list(TASK_PROMPTS.keys()),
                        value="Text",
                        label="Recognition task",
                    )
                    max_toks = gr.Slider(
                        minimum=64,
                        maximum=4096,
                        value=1024,
                        step=64,
                        label="max_new_tokens",
                    )
                    run_btn = gr.Button("Recognize", variant="primary")
                with gr.Column(scale=1):
                    output = _result_textbox("Result")
            run_btn.click(run_single, inputs=[image_in, task, max_toks], outputs=output)

            available_examples = [[str(p), task_name, 1024] for p, task_name in example_files if Path(p).exists()]
            if available_examples:
                gr.Examples(
                    examples=available_examples,
                    inputs=[image_in, task, max_toks],
                    label="Examples",
                )

        if detector is not None:
            with gr.Tab("Full document parsing (PP-DocLayoutV3 + GLM-OCR)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        doc_in = gr.Image(type="pil", label="Document image")
                        doc_max = gr.Slider(
                            minimum=64,
                            maximum=2048,
                            value=512,
                            step=64,
                            label="max_new_tokens",
                        )
                        doc_thr = gr.Slider(
                            minimum=0.05,
                            maximum=0.9,
                            value=0.3,
                            step=0.05,
                            label="score_threshold",
                        )
                        doc_btn = gr.Button("Parse document", variant="primary")
                    with gr.Column(scale=1):
                        layout_preview = gr.Image(type="pil", label="Detected regions")
                        doc_output = _result_textbox("Markdown result", lines=24)
                doc_btn.click(
                    run_pipeline,
                    inputs=[doc_in, doc_max, doc_thr],
                    outputs=[layout_preview, doc_output],
                )
                doc_example = Path("ocr_doc_sample.png")
                if doc_example.exists():
                    gr.Examples(
                        examples=[[str(doc_example), 512, 0.3]],
                        inputs=[doc_in, doc_max, doc_thr],
                        label="Examples",
                    )

    return demo


def download_example_assets() -> list[Path]:
    """Download example images from their canonical GitHub user-attachment URLs.

    Uses :func:`notebook_utils.download_file`, which already handles the
    existence check, progress bar, retries and User-Agent header. Must be
    called from the notebook directory (where ``notebook_utils.py`` has been
    fetched into).
    """
    from notebook_utils import download_file

    return [download_file(url, filename=name) for name, url in EXAMPLE_ASSET_URLS.items()]
