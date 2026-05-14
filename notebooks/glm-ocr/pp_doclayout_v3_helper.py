"""Helpers for running PP-DocLayoutV3 layout detection through OpenVINO.

PP-DocLayoutV3 is a PaddleOCR layout-detection model that complements
GLM-OCR: it splits a document image into region-level crops and classifies
each region (text / title / table / formula / ...). GLM-OCR is then invoked
with the prompt matching each region's class.

This module keeps runtime dependencies minimal — PaddlePaddle is **not**
required. We use ``PaddlePaddle/PP-DocLayoutV3_safetensors`` from
HuggingFace, which ships a native ``transformers`` implementation of the
model. The PyTorch weights are exported to ONNX once and then run by
``openvino.Core().compile_model(onnx_path, device)`` — no IR conversion
step is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"


# Classes shared across detections. Populated dynamically from
# ``model.config.id2label`` when the ONNX is exported; we fall back to this
# hard-coded map so the helper can still run if the config is unreadable.
# Duplicates in ``id2label`` are intentional — upstream uses them to group
# multiple internal heads under the same display label.
DEFAULT_ID2LABEL: Dict[int, str] = {
    0: "abstract",
    1: "algorithm",
    2: "aside_text",
    3: "chart",
    4: "content",
    5: "formula",
    6: "doc_title",
    7: "figure_title",
    8: "footer",
    9: "footer",
    10: "footnote",
    11: "formula_number",
    12: "header",
    13: "header",
    14: "image",
    15: "formula",
    16: "number",
    17: "paragraph_title",
    18: "reference",
    19: "reference_content",
    20: "seal",
    21: "table",
    22: "text",
    23: "text",
    24: "vision_footnote",
}


PROMPT_BY_CLASS: Dict[str, str] = {
    "text": "Text Recognition:",
    "paragraph_title": "Text Recognition:",
    "doc_title": "Text Recognition:",
    "figure_title": "Text Recognition:",
    "table_title": "Text Recognition:",
    "chart_title": "Text Recognition:",
    "abstract": "Text Recognition:",
    "content": "Text Recognition:",
    "reference": "Text Recognition:",
    "reference_content": "Text Recognition:",
    "footnote": "Text Recognition:",
    "header": "Text Recognition:",
    "footer": "Text Recognition:",
    "aside_text": "Text Recognition:",
    "number": "Text Recognition:",
    "algorithm": "Text Recognition:",
    "vision_footnote": "Text Recognition:",
    "formula": "Formula Recognition:",
    "formula_number": "Formula Recognition:",
    "table": "Table Recognition:",
}


# ---------------------------------------------------------------------------
# Export PyTorch -> ONNX (one-off, cached in ``output_dir``)
# ---------------------------------------------------------------------------


class _Wrapper:
    """Strip the transformers output dataclass down to ``(logits, pred_boxes)``.

    Defined lazily inside :func:`export_pp_doclayout_v3` so ``torch`` is only
    imported when a conversion is actually requested.
    """


def export_pp_doclayout_v3(
    output_dir: str | Path = "pp_doclayout_v3_ov",
    model_id: str = MODEL_ID,
    opset_version: int = 17,
) -> Path:
    """Download PP-DocLayoutV3 and export it to ONNX.

    Returns the directory holding ``pp_doclayout_v3.onnx`` plus any sidecar
    weight files (``.onnx.data`` when PyTorch emits external tensors).
    """
    import inspect

    import torch
    from transformers import AutoModelForObjectDetection

    output_dir = Path(output_dir)
    onnx_path = output_dir / "pp_doclayout_v3.onnx"
    if onnx_path.exists():
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForObjectDetection.from_pretrained(model_id).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            out = self.m(pixel_values=pixel_values)
            return out.logits, out.pred_boxes

    wrapped = Wrapper(model).eval()
    dummy = torch.randn(1, 3, 800, 800)

    # Force the legacy TorchScript exporter. torch>=2.5 routes through
    # onnxscript by default, which on torch 2.11 emits modern Pad ops and
    # then asks onnx.version_converter to downgrade them to ``opset_version``
    # — the downgrader has no adapter for Pad, so the whole export aborts
    # with ``No Adapter To Version $17 for Pad``. The legacy exporter does
    # not trip this path.
    export_kwargs = dict(
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=False,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(wrapped, (dummy,), onnx_path, **export_kwargs)

    # Persist id2label for the detector so it does not need a second model
    # download at inference time.
    import json

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    (output_dir / "id2label.json").write_text(json.dumps(id2label, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class LayoutDetector:
    """Thin OpenVINO wrapper for the exported PP-DocLayoutV3 ONNX model.

    Loads the ONNX file directly via ``ov.Core().compile_model`` — no
    intermediate IR step — and reuses the HuggingFace
    ``PPDocLayoutV3ImageProcessor`` for preprocessing.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "CPU",
        model_id: str = MODEL_ID,
    ):
        import json

        import openvino as ov
        from transformers import AutoImageProcessor

        model_dir = Path(model_dir)
        onnx_path = model_dir / "pp_doclayout_v3.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"{onnx_path} not found. Run export_pp_doclayout_v3() first.")

        self.model = ov.Core().compile_model(str(onnx_path), device)
        self.input_name = self.model.inputs[0].get_any_name()

        # Preprocessor: 800x800 resize, /255 rescale, no normalization.
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        id2label_path = model_dir / "id2label.json"
        if id2label_path.exists():
            self.id2label = {int(k): v for k, v in json.loads(id2label_path.read_text()).items()}
        else:
            self.id2label = DEFAULT_ID2LABEL

    # -- preprocessing / postprocessing ------------------------------------

    def preprocess(self, image):
        from PIL import Image as _Image, ImageOps

        if not isinstance(image, _Image.Image):
            image = _Image.open(image)
        # Apply EXIF orientation so the preprocessor and bbox post-scaling
        # both see the image in its visual orientation. Without this,
        # phone-camera JPEGs with an orientation tag produce detections
        # computed on one orientation and scaled against the raw ``Image.size``
        # of another — boxes land in the wrong place.
        image = ImageOps.exif_transpose(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="np")
        return inputs["pixel_values"], image.size, image  # (W, H), corrected PIL

    def __call__(self, image, score_thr: float = 0.3) -> List[Dict[str, Any]]:
        pixel_values, (w, h), _ = self.preprocess(image)
        result = self.model({self.input_name: pixel_values})
        logits = list(result.values())[0][0]  # (num_queries, num_classes)
        boxes = list(result.values())[1][0]  # (num_queries, 4) in cx,cy,w,h (0..1)

        # Sigmoid + flat top-k (mirrors PPDocLayoutV3ImageProcessor)
        scores = _sigmoid(logits)  # (Q, C)
        flat = scores.flatten()
        num_classes = scores.shape[1]
        topk = scores.shape[0]
        idx = np.argpartition(-flat, topk - 1)[:topk]
        idx = idx[np.argsort(-flat[idx])]
        scr = flat[idx]
        lbl = idx % num_classes
        qry = idx // num_classes

        # Centre/size -> xyxy; scale to original image size
        cb = boxes[qry]
        xyxy = np.concatenate([cb[:, :2] - 0.5 * cb[:, 2:], cb[:, :2] + 0.5 * cb[:, 2:]], axis=-1)
        xyxy = xyxy * np.array([w, h, w, h], dtype=xyxy.dtype)

        keep = scr >= score_thr
        detections: List[Dict[str, Any]] = []
        for s, l, b in zip(scr[keep].tolist(), lbl[keep].tolist(), xyxy[keep].tolist()):
            cls_name = self.id2label.get(int(l))
            if cls_name is None:
                continue
            x0, y0, x1, y1 = b
            x0, x1 = float(max(0.0, min(w, x0))), float(max(0.0, min(w, x1)))
            y0, y1 = float(max(0.0, min(h, y0))), float(max(0.0, min(h, y1)))
            if x1 <= x0 or y1 <= y0:
                continue
            detections.append(
                {
                    "bbox": [x0, y0, x1, y1],
                    "class": cls_name,
                    "score": float(s),
                    "label_id": int(l),
                }
            )

        # Reading-order heuristic: top-to-bottom, then left-to-right within a row.
        detections.sort(key=lambda d: (round(d["bbox"][1] / 20.0), d["bbox"][0]))
        return detections


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


_DEFAULT_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (255, 99, 132),
    (54, 162, 235),
    (255, 206, 86),
    (75, 192, 192),
    (153, 102, 255),
    (255, 159, 64),
    (199, 199, 199),
    (83, 102, 255),
    (255, 99, 255),
    (99, 255, 132),
)


def draw_layout(image, detections: List[Dict[str, Any]], output_path: str | Path | None = None):
    """Draw detection boxes with labels on ``image``.

    Returns the annotated PIL image (optionally also written to ``output_path``).
    """
    from PIL import Image as _Image, ImageDraw, ImageFont, ImageOps

    if not isinstance(image, _Image.Image):
        image = _Image.open(image)
    image = ImageOps.exif_transpose(image).convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    class_colors: Dict[str, Tuple[int, int, int]] = {}
    for det in detections:
        cls = det["class"]
        if cls not in class_colors:
            class_colors[cls] = _DEFAULT_PALETTE[len(class_colors) % len(_DEFAULT_PALETTE)]
        colour = class_colors[cls]
        x0, y0, x1, y1 = det["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=2)
        label = f"{cls} {det['score']:.2f}"
        try:
            left, top, right, bottom = draw.textbbox((x0, max(0.0, y0 - 16)), label, font=font)
            draw.rectangle([left - 1, top - 1, right + 1, bottom + 1], fill=colour)
        except AttributeError:
            pass
        draw.text((x0 + 2, max(0.0, y0 - 15)), label, fill=(255, 255, 255), font=font)

    if output_path is not None:
        image.save(str(output_path))
    return image


# ---------------------------------------------------------------------------
# End-to-end pipeline (PP-DocLayoutV3 -> GLM-OCR)
# ---------------------------------------------------------------------------


def _format_region_markdown(cls: str, text: str) -> str:
    """Wrap ``text`` with the Markdown fencing appropriate for ``cls``."""
    if cls == "doc_title":
        return f"# {text}"
    if cls in {"paragraph_title", "figure_title", "table_title", "chart_title"}:
        return f"## {text}"
    if cls in {"formula", "formula_number"}:
        return f"$$\n{text}\n$$"
    return text


def iter_pipeline(
    detector: LayoutDetector,
    ocr_model,
    processor,
    image,
    max_new_tokens: int = 1024,
    score_thr: float = 0.3,
):
    """Stream PP-DocLayoutV3 + GLM-OCR recognition region-by-region.

    Yields ``(det, chunk, final_markdown)`` tuples:

    - ``det`` is the current detection dict (``class`` / ``score`` / ``bbox``).
    - ``chunk`` is the newly-decoded text fragment for this region (one item
      per call to the underlying :class:`~transformers.TextIteratorStreamer`).
      ``None`` when the region has no matching prompt.
    - ``final_markdown`` is non-``None`` on the region's final yield and
      contains the full Markdown-formatted recognition for that region.
    """
    from threading import Thread

    from PIL import Image as _Image, ImageOps
    from transformers import TextIteratorStreamer

    if not isinstance(image, _Image.Image):
        image = _Image.open(image)
    image = ImageOps.exif_transpose(image).convert("RGB")

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    detections = detector(image, score_thr=score_thr)
    for det in detections:
        cls = det["class"]
        prompt = PROMPT_BY_CLASS.get(cls)
        x0, y0, x1, y1 = det["bbox"]
        if prompt is None:
            placeholder = f"<!-- region: {cls} ({x0:.0f},{y0:.0f}-{x1:.0f},{y1:.0f}) -->"
            yield det, None, placeholder
            continue

        crop = image.crop((x0, y0, x1, y1))
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

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
        )
        thread = Thread(target=ocr_model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        buf = ""
        for piece in streamer:
            buf += piece
            yield det, piece, None
        thread.join(timeout=1.0)

        yield det, None, _format_region_markdown(cls, buf.strip())


def run_pipeline(
    detector: LayoutDetector,
    ocr_model,
    processor,
    image,
    max_new_tokens: int = 1024,
    score_thr: float = 0.3,
) -> str:
    """Run PP-DocLayoutV3 + GLM-OCR as a single document parser.

    Returns a Markdown string composed of per-region recognition outputs,
    ordered by the layout reading-order heuristic in ``LayoutDetector``.
    """
    parts: List[str] = []
    for _det, _chunk, final in iter_pipeline(
        detector,
        ocr_model,
        processor,
        image,
        max_new_tokens=max_new_tokens,
        score_thr=score_thr,
    ):
        if final is not None:
            parts.append(final)
    return "\n\n".join(parts)
