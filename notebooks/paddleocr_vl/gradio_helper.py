"""
Gradio demo for PaddleOCR-VL OpenVINO pipeline.

This file is intentionally lightweight and follows the inference flow from:
`paddleocr_vl.ipynb` (cell "Validate the end-to-end OpenVINO pipeline").

Usage in notebook:
    from gradio_helper import make_demo
    demo = make_demo(paddleocr_vl_model)
    demo.launch(debug=True)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
from PIL import Image, ImageDraw, ImageOps

# Keep prompts aligned with the notebook (Cell 13).
PROMPTS: Dict[str, str] = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


def _ensure_rgb(pil_image: Image.Image) -> Image.Image:
    """Convert to RGB and apply EXIF transpose for correct orientation."""
    pil_image = ImageOps.exif_transpose(pil_image)
    if pil_image.mode in ("RGBA", "LA", "P"):
        pil_image = pil_image.convert("RGB")
    elif pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image


def _build_messages(pil_image: Image.Image, task: str) -> list:
    prompt_text = PROMPTS.get(task, PROMPTS["ocr"])
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def _build_generation_config(model: Any, max_new_tokens: int) -> Dict[str, Any]:
    # The notebook reads token ids from `model.tokenizer`.
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("model.tokenizer is required (missing tokenizer on the provided model).")
    return {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
    }


def _get_or_create_sample_image(sample_path: Path) -> Optional[Path]:
    """
    Best-effort helper: create a minimal sample image if it does not exist.
    This keeps the demo self-contained, similar to the notebook.
    """
    try:
        if sample_path.exists():
            return sample_path
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        im = Image.new("RGB", (300, 200), color=(255, 255, 255))
        d = ImageDraw.Draw(im)
        d.text(
            (40, 40),
            "PaddleOCR-VL OpenVINO demo\nOCR: Hello 123\nTable: A | B | C",
            fill=(0, 0, 0),
        )
        im.save(sample_path)
        return sample_path
    except Exception:
        return None


def make_demo(paddleocr_vl_model: Any) -> gr.Blocks:
    """
    Create a Gradio Blocks demo.

    Requirements:
    - paddleocr_vl_model must provide: .chat(messages=..., generation_config=...)
    - paddleocr_vl_model must provide: .tokenizer with bos/eos/pad token ids
    """

    def run(
        image: Optional[Image.Image],
        task: str,
        max_new_tokens: int,
    ) -> Tuple[str, str, Dict[str, Any]]:
        if image is None:
            return "Please upload an image first.", "", {"error": "missing_image"}

        t0 = time.perf_counter()
        pil_image = _ensure_rgb(image)

        messages = _build_messages(pil_image, task)
        try:
            generation_config = _build_generation_config(paddleocr_vl_model, max_new_tokens=max_new_tokens)
            t1 = time.perf_counter()
            response, _history = paddleocr_vl_model.chat(messages=messages, generation_config=generation_config)
            t2 = time.perf_counter()
        except Exception as e:
            return (
                f"Inference failed: {type(e).__name__}: {e}",
                "",
                {
                    "task": task,
                    "max_new_tokens": int(max_new_tokens),
                    "messages": _safe_serialize_messages(messages),
                    "error": repr(e),
                },
            )

        text = (response or "").strip()
        md = f"```text\n{text}\n```" if text else ""
        debug = {
            "task": task,
            "max_new_tokens": int(max_new_tokens),
            "messages": _safe_serialize_messages(messages),
            "generation_config": generation_config,
            "timing_ms": {
                "preprocess": (t1 - t0) * 1000.0,
                "chat": (t2 - t1) * 1000.0,
                "total": (t2 - t0) * 1000.0,
            },
        }
        return text, md, debug

    # Sample image (optional).
    sample_path = _get_or_create_sample_image(Path(__file__).parent / "test.png")
    examples = [[str(sample_path)]] if sample_path else []

    with gr.Blocks(title="PaddleOCR-VL (OpenVINO)") as demo:
        gr.Markdown("""
## PaddleOCR-VL (OpenVINO) Interactive Demo

- Supported tasks: OCR / Table / Formula / Chart 
""")

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(label="Input Image", type="pil", height=800)
                task_in = gr.Dropdown(
                    choices=list(PROMPTS.keys()),
                    value="ocr",
                    label="Task",
                )
                max_new_tokens_in = gr.Slider(
                    minimum=16,
                    maximum=2048,
                    value=512,
                    step=1,
                    label="max_new_tokens",
                )
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📝 Text"):
                        text_out = gr.Textbox(lines=20, show_label=False)
                    with gr.Tab("📋 Markdown"):
                        md_out = gr.Markdown("")
                    with gr.Tab("🔍 Debug"):
                        debug_out = gr.JSON(label="messages / generation_config / timing")

        if examples:
            gr.Examples(examples=examples, inputs=[image_in], label="Examples")

        run_btn.click(
            fn=run,
            inputs=[image_in, task_in, max_new_tokens_in],
            outputs=[text_out, md_out, debug_out],
        )

    return demo


def _safe_serialize_messages(messages: list) -> list:
    """
    Convert messages to JSON-friendly form (PIL.Image is not serializable).
    We keep prompt text and replace image with a short descriptor.
    """
    safe = []
    for msg in messages:
        msg2 = {"role": msg.get("role"), "content": []}
        for c in msg.get("content", []):
            if c.get("type") == "image":
                img = c.get("image", None)
                if isinstance(img, Image.Image):
                    msg2["content"].append(
                        {
                            "type": "image",
                            "image": {
                                "mode": img.mode,
                                "size": list(img.size),
                            },
                        }
                    )
                else:
                    msg2["content"].append({"type": "image", "image": "<non-pil-image>"})
            else:
                msg2["content"].append({"type": c.get("type"), "text": c.get("text", "")})
        safe.append(msg2)
    return safe
