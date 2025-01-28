from typing import Callable, Tuple
import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import requests

examples = [
    ["https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg"],
    ["https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/empty_road_mapillary.jpg"],
    ["https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/wall.jpg"],
]

# last_image = examples[0][0]

example_images = []
for example_image_url in examples:
    image_name = example_image_url[0].split("/")[-1]
    if not Path(image_name).exists():
        image = Image.open(requests.get(example_image_url[0], stream=True).raw)
        image.save(image_name)
    example_images.append([image_name])

last_image = example_images[0][0]


def select_point(img: Image.Image, point_type: str, evt: gr.SelectData) -> Image.Image:
    """Gradio select callback."""
    img = img.convert("RGBA")
    x, y = evt.index[0], evt.index[1]
    point_radius = np.round(max(img.size) / 100)
    if point_type == "Object point":
        object_points.append((x, y))
        color = (30, 255, 30, 200)
    elif point_type == "Background point":
        background_points.append((x, y))
        color = (255, 30, 30, 200)
    elif point_type == "Bounding Box":
        bbox_points.append((x, y))
        color = (10, 10, 255, 255)
        if len(bbox_points) % 2 == 0:
            # Draw a rectangle if number of points is even
            new_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
            _draw = ImageDraw.Draw(new_img)
            x0, y0, x1, y1 = *bbox_points[-2], *bbox_points[-1]
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            # Save sorted order
            bbox_points[-2] = (x0, y0)
            bbox_points[-1] = (x1, y1)
            _draw.rectangle((x0, y0, x1, y1), fill=(*color[:-1], 90))
            img = Image.alpha_composite(img, new_img)
    # Draw a point
    ImageDraw.Draw(img).ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=color,
    )
    return img


def clear_points() -> Tuple[Image.Image, None]:
    """Gradio clear points callback."""
    global object_points, background_points, bbox_points
    # global object_points; global background_points; global bbox_points
    object_points = []
    background_points = []
    bbox_points = []
    return last_image, None


def save_last_picked_image(img: Image.Image) -> None:
    """Gradio callback saves the last used image."""
    global last_image
    last_image = img
    # If we change the input image
    # we should clear all the previous points
    clear_points()
    # Removes the segmentation map output
    return None


def make_demo(fn: Callable, quantized: bool):
    with gr.Blocks(title="Fast SAM") as demo:
        with gr.Row(variant="panel"):
            original_img = gr.Image(label="Input", type="pil")
            segmented_img = gr.Image(label="Segmentation Map", type="pil")
        with gr.Row():
            point_type = gr.Radio(
                ["Object point", "Background point", "Bounding Box"],
                value="Object point",
                label="Pixel selector type",
            )
            model_type = gr.Radio(
                ["FP32 model", "Quantized model"] if quantized else ["FP32 model"],
                value="FP32 model",
                label="Select model variant",
            )
        with gr.Row(variant="panel"):
            segment_button = gr.Button("Segment", variant="primary")
            clear_button = gr.Button("Clear points", variant="secondary")
        gr.Examples(
            example_images,
            inputs=original_img,
            fn=save_last_picked_image,
            run_on_click=True,
            outputs=segmented_img,
        )

        # Callbacks
        original_img.select(select_point, inputs=[original_img, point_type], outputs=original_img)
        original_img.upload(save_last_picked_image, inputs=original_img, outputs=segmented_img)
        clear_button.click(clear_points, outputs=[original_img, segmented_img])
        segment_button.click(fn, inputs=[original_img, model_type], outputs=segmented_img)
    return demo
