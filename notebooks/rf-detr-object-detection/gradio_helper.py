from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageOps


def run_object_detection(
    model,
    processor,
    image: Image.Image,
    threshold: float,
):
    image = ImageOps.exif_transpose(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_object_detection(
        outputs=outputs,
        threshold=threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    visualization = image.copy()
    draw = ImageDraw.Draw(visualization)
    detections = []

    for score, label, box in zip(
        result["scores"],
        result["labels"],
        result["boxes"],
    ):
        left, top, right, bottom = (int(round(float(coordinate))) for coordinate in box)
        left = max(0, min(left, image.width - 1))
        top = max(0, min(top, image.height - 1))
        right = max(left, min(right, image.width - 1))
        bottom = max(top, min(bottom, image.height - 1))
        label_id = int(label)
        id2label = getattr(model.config, "id2label", {})
        label_name = id2label.get(label_id, str(label_id))
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


def make_demo(model, processor, example_image: str | Path | None = None):
    def detect(image, threshold):
        if image is None:
            return None, []
        return run_object_detection(model, processor, image, float(threshold))

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
