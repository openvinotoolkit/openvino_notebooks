from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from rfdetr.assets.coco_classes import COCO_CLASSES
from torchvision.transforms import functional as F


def run_object_detection(
    model,
    image: Image.Image,
    threshold: float,
):
    image = ImageOps.exif_transpose(image).convert("RGB")
    input_shape = model.input(0).shape
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] != 3:
        raise ValueError(f"Expected a static NCHW RGB model with batch size 1, got {input_shape}")
    input_height, input_width = input_shape[2:]
    image_tensor = F.to_tensor(image)
    image_tensor = F.resize(image_tensor, [input_height, input_width], antialias=False)
    image_tensor = F.normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_array = np.ascontiguousarray(image_tensor.unsqueeze(0).numpy(), dtype=np.float32)

    if len(model.outputs) != 2:
        raise ValueError(f"Expected boxes and logits, got {len(model.outputs)} outputs")
    request = model.create_infer_request()
    request.infer({model.input(0): input_array})
    boxes = request.get_output_tensor(0).data
    logits = request.get_output_tensor(1).data
    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[-1] != 4 or logits.ndim != 3 or logits.shape[:2] != boxes.shape[:2]:
        raise ValueError(f"Unexpected RF-DETR output shapes: boxes={boxes.shape}, logits={logits.shape}")

    scores = 1.0 / (1.0 + np.exp(-np.clip(logits[0], -88, 88)))
    flat_scores = scores.reshape(-1)
    top_indices = np.argsort(-flat_scores, kind="stable")[: boxes.shape[1]]
    top_indices = top_indices[flat_scores[top_indices] > threshold]
    query_indices, class_ids = np.divmod(top_indices, scores.shape[1])
    selected_boxes = boxes[0, query_indices]

    visualization = image.copy()
    draw = ImageDraw.Draw(visualization)
    detections = []

    for score, label_id, (cx, cy, width, height) in zip(flat_scores[top_indices], class_ids, selected_boxes):
        coordinates = (
            (cx - width / 2) * image.width,
            (cy - height / 2) * image.height,
            (cx + width / 2) * image.width,
            (cy + height / 2) * image.height,
        )
        left, top, right, bottom = (round(float(coordinate)) for coordinate in coordinates)
        left = max(0, min(left, image.width - 1))
        top = max(0, min(top, image.height - 1))
        right = max(left, min(right, image.width - 1))
        bottom = max(top, min(bottom, image.height - 1))
        label_id = int(label_id)
        label_name = COCO_CLASSES.get(label_id, str(label_id))
        confidence = float(score)
        caption = f"{label_name}: {confidence:.2f}"

        draw.rectangle((left, top, right, bottom), outline="#00A3A3", width=3)
        text_box = draw.textbbox((0, 0), caption)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_top = top - text_height - 6 if top >= text_height + 6 else top
        text_right = min(image.width - 1, left + text_width + 6)
        text_bottom = min(image.height - 1, text_top + text_height + 6)
        draw.rectangle(
            (left, text_top, text_right, text_bottom),
            fill="#00A3A3",
        )
        draw.text((left + 3, text_top + 3), caption, fill="white")

        detections.append(
            {
                "label": label_name,
                "score": round(confidence, 4),
                "box": {
                    "xmin": left,
                    "ymin": top,
                    "xmax": right,
                    "ymax": bottom,
                },
            }
        )

    return visualization, detections


def make_demo(model, example_image: str | Path | None = None):
    def detect(image, threshold):
        if image is None:
            return None, []
        return run_object_detection(model, image, float(threshold))

    with gr.Blocks(title="RF-DETR Object Detection with OpenVINO") as demo:
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input image")
                threshold = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=0.4,
                    step=0.05,
                    label="Confidence threshold",
                )
                detect_button = gr.Button("Detect objects", variant="primary")
            with gr.Column():
                image_output = gr.Image(type="pil", label="Detections")
                detections_output = gr.JSON(label="Detection data")

        detect_button.click(
            detect,
            inputs=[image_input, threshold],
            outputs=[image_output, detections_output],
        )

        if example_image is not None and Path(example_image).exists():
            gr.Examples(
                examples=[[str(example_image), 0.4]],
                inputs=[image_input, threshold],
            )

    return demo
