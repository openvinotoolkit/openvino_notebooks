"""Helpers for running PP-DocLayout-V3 layout detection on OpenVINO.

PP-DocLayout-V3 is a PaddlePaddle layout-detection model that complements
GLM-OCR: it splits a document image into region-level crops and classifies
each region (text / title / table / formula / ...). GLM-OCR is then invoked
with the prompt matching each region's class.

This module stays intentionally lightweight so it can be imported from the
GLM-OCR notebook without pulling in PaddlePaddle at runtime — only the
conversion helper ``convert_pp_doclayout_v3`` depends on ``paddle2onnx``.
"""

from __future__ import annotations

import subprocess  # nosec B404 - used for paddle2onnx CLI only
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


DEFAULT_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/"
    "paddle3.0.0/PP-DocLayout_plus-L_infer.tar"
)


LAYOUT_CLASSES = [
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "table_title",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
    "chart_title",
    "chart",
    "formula_number",
    "header_image",
    "footer_image",
    "aside_text",
]


PROMPT_BY_CLASS = {
    "text": "Text Recognition:",
    "paragraph_title": "Text Recognition:",
    "doc_title": "Text Recognition:",
    "figure_title": "Text Recognition:",
    "table_title": "Text Recognition:",
    "chart_title": "Text Recognition:",
    "abstract": "Text Recognition:",
    "content": "Text Recognition:",
    "reference": "Text Recognition:",
    "footnote": "Text Recognition:",
    "header": "Text Recognition:",
    "footer": "Text Recognition:",
    "aside_text": "Text Recognition:",
    "number": "Text Recognition:",
    "algorithm": "Text Recognition:",
    "formula": "Formula Recognition:",
    "formula_number": "Formula Recognition:",
    "table": "Table Recognition:",
}


# ---------------------------------------------------------------------------
# Conversion (Paddle -> ONNX -> OpenVINO IR)
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # nosec B310 - known Paddle CDN


def _extract(tar_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(dest)  # nosec B202 - trusted upstream archive
    subs = [p for p in dest.iterdir() if p.is_dir()]
    return subs[0] if subs else dest


def _paddle2onnx(pd_dir: Path, onnx_path: Path) -> None:
    if onnx_path.exists():
        return
    cmd = [
        sys.executable,
        "-m",
        "paddle2onnx",
        "--model_dir",
        str(pd_dir),
        "--model_filename",
        "inference.pdmodel",
        "--params_filename",
        "inference.pdiparams",
        "--save_file",
        str(onnx_path),
        "--opset_version",
        "16",
        "--enable_onnx_checker",
        "True",
    ]
    subprocess.check_call(cmd)  # nosec B603 - fully-quoted paddle2onnx invocation


def convert_pp_doclayout_v3(
    output_dir: str | Path = "pp_doclayout_v3_ov",
    url: str = DEFAULT_URL,
    work_dir: str | Path = "pp_doclayout_v3_src",
) -> Path:
    """Download PP-DocLayout-V3 and convert it to OpenVINO IR.

    Returns the directory containing ``pp_doclayout_v3.xml``.
    """
    import openvino as ov

    output_dir = Path(output_dir)
    work_dir = Path(work_dir)

    if (output_dir / "pp_doclayout_v3.xml").exists():
        return output_dir

    work_dir.mkdir(parents=True, exist_ok=True)
    tar_path = work_dir / Path(url).name
    _download(url, tar_path)
    pd_dir = _extract(tar_path, work_dir / "extracted")
    onnx_path = work_dir / "model.onnx"
    _paddle2onnx(pd_dir, onnx_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, output_dir / "pp_doclayout_v3.xml", compress_to_fp16=True)
    return output_dir


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class LayoutDetector:
    """Thin OV wrapper for PP-DocLayout-V3.

    Expects a model exported via :func:`convert_pp_doclayout_v3`.
    """

    def __init__(self, model_dir: str | Path, device: str = "CPU", input_size: int = 800):
        import openvino as ov

        xml = next(Path(model_dir).glob("*.xml"))
        self.model = ov.Core().compile_model(str(xml), device)
        self.input_name = self.model.inputs[0].get_any_name()
        self.input_size = input_size

    def preprocess(self, image):
        from PIL import Image as _Image

        img = image.convert("RGB") if isinstance(image, _Image.Image) else _Image.open(image).convert("RGB")
        orig_w, orig_h = img.size
        scale = self.input_size / max(orig_w, orig_h)
        resized = img.resize((int(orig_w * scale), int(orig_h * scale)))
        canvas = _Image.new("RGB", (self.input_size, self.input_size), (127, 127, 127))
        canvas.paste(resized, (0, 0))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return arr.transpose(2, 0, 1)[None], scale, (orig_w, orig_h)

    def __call__(self, image, score_thr: float = 0.3) -> List[Dict[str, Any]]:
        inp, scale, (w, h) = self.preprocess(image)
        outputs = self.model({self.input_name: inp})
        main_output = list(outputs.values())[0]
        if main_output.ndim == 3:
            main_output = main_output[0]
        detections = []
        for row in main_output:
            if row.shape[0] < 6:
                continue
            x0, y0, x1, y1, cls, score = [float(v) for v in row[:6]]
            if score < score_thr:
                continue
            x0, x1 = x0 / scale, x1 / scale
            y0, y1 = y0 / scale, y1 / scale
            x0 = max(0.0, min(w, x0))
            x1 = max(0.0, min(w, x1))
            y0 = max(0.0, min(h, y0))
            y1 = max(0.0, min(h, y1))
            cls_idx = int(cls)
            if cls_idx < 0 or cls_idx >= len(LAYOUT_CLASSES):
                continue
            detections.append(
                {"bbox": [x0, y0, x1, y1], "class": LAYOUT_CLASSES[cls_idx], "score": score}
            )
        detections.sort(key=lambda d: (round(d["bbox"][1] / 20.0), d["bbox"][0]))
        return detections


def run_pipeline(
    detector: LayoutDetector,
    ocr_model,
    processor,
    image,
    max_new_tokens: int = 1024,
    score_thr: float = 0.3,
) -> str:
    """Run PP-DocLayout-V3 + GLM-OCR as a single document parser.

    Returns a Markdown string composed of per-region recognition outputs.
    """
    from PIL import Image as _Image

    if not isinstance(image, _Image.Image):
        image = _Image.open(image).convert("RGB")

    detections = detector(image, score_thr=score_thr)
    parts: List[str] = []
    for i, det in enumerate(detections):
        cls = det["class"]
        prompt = PROMPT_BY_CLASS.get(cls)
        x0, y0, x1, y1 = det["bbox"]
        crop = image.crop((x0, y0, x1, y1))
        if prompt is None:
            parts.append(f"![{cls} region {i}]()")
            continue
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": prompt},
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
        gen = ocr_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.decode(gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        if cls == "doc_title":
            parts.append(f"# {text}")
        elif cls in {"paragraph_title", "figure_title", "table_title", "chart_title"}:
            parts.append(f"## {text}")
        elif cls == "formula":
            parts.append(f"$$\n{text}\n$$")
        else:
            parts.append(text)
    return "\n\n".join(parts)
