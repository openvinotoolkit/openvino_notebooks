"""Minimal Gradio demo for MinerU 2.5 with OpenVINO backend.

The official MinerU Hugging Face Space ships a feature-heavy front-end that
relies on the full ``mineru`` pipeline (queueing, multiple backends, model
download UI, etc.). For a self-contained notebook demo, this helper builds a
tiny side-by-side viewer: upload a PDF or image on the left, see the parsed
Markdown rendered on the right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gradio as gr

from ov_mineru_helper import OVMinerUClient, pdf_to_images

_EXAMPLE_PDF_URL = "https://raw.githubusercontent.com/opendatalab/MinerU/master/demo/pdfs/demo1.pdf"


def _download_example(target_dir: Path) -> Optional[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "demo1.pdf"
    if path.exists():
        return path
    try:
        import requests

        resp = requests.get(_EXAMPLE_PDF_URL, timeout=30)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        return path
    except Exception:
        return None


def make_demo(client: OVMinerUClient) -> gr.Blocks:
    example_pdf = _download_example(Path("examples"))

    def _parse(file, dpi, image_analysis, progress=gr.Progress(track_tqdm=False)):
        if file is None:
            return "*Please upload a PDF or image first.*", "", None
        path = Path(file if isinstance(file, str) else file.name)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            pages = pdf_to_images(path, dpi=int(dpi))
        else:
            from PIL import Image

            pages = [Image.open(path).convert("RGB")]

        md_chunks = []
        for idx, page in enumerate(pages):
            progress((idx) / max(len(pages), 1), desc=f"Parsing page {idx + 1}/{len(pages)}")
            md, _ = client.image_to_markdown(page, image_analysis=bool(image_analysis))
            md_chunks.append(f"<!-- page {idx + 1} -->\n\n{md}")
        progress(1.0, desc="Done")

        markdown = "\n\n---\n\n".join(md_chunks)
        preview = pages[0]
        return markdown, markdown, preview

    with gr.Blocks(title="MinerU 2.5 · OpenVINO") as demo:
        gr.Markdown(
            "# MinerU 2.5 with OpenVINO\n"
            "Upload a **PDF** or **image** of a document page and get a Markdown "
            "transcription produced by `MinerU2.5-Pro-2604-1.2B` running on "
            "OpenVINO."
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(
                    label="PDF or image",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"],
                )
                dpi = gr.Slider(96, 300, value=200, step=4, label="PDF render DPI")
                image_analysis = gr.Checkbox(
                    value=False,
                    label="Enable image / chart analysis (slower)",
                )
                run_btn = gr.Button("Parse document", variant="primary")
                first_page = gr.Image(label="First page preview", interactive=False)
                if example_pdf is not None:
                    gr.Examples(
                        examples=[[str(example_pdf)]],
                        inputs=[file_in],
                        label="Example",
                    )
            with gr.Column(scale=2):
                with gr.Tab("Rendered"):
                    md_out = gr.Markdown(label="Markdown")
                with gr.Tab("Source"):
                    md_raw = gr.Code(label="Markdown source", language="markdown")

        run_btn.click(
            _parse,
            inputs=[file_in, dpi, image_analysis],
            outputs=[md_out, md_raw, first_page],
        )

    return demo
