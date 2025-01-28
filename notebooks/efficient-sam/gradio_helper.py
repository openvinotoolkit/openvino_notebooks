from pathlib import Path
from typing import Callable
import gradio as gr
import requests
import copy
from PIL import ImageDraw


def get_examples():
    example_images = [
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/b8083dd5-1ce7-43bf-8b09-a2ebc280c86e",
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/9a90595d-70e7-469b-bdaf-469ef4f56fa2",
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/b626c123-9fa2-4aa6-9929-30565991bf0c",
    ]
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    for img_id, image_url in enumerate(example_images):
        img_path = examples_dir / f"example_{img_id}.jpg"
        if not img_path.exists():
            r = requests.get(image_url)
            with img_path.open("wb") as f:
                f.write(r.content)
    return [[img] for img in examples_dir.glob("*.jpg")]


examples = get_examples()

default_example = examples[0]

title = "<center><strong><font size='8'>Efficient Segment Anything with OpenVINO and EfficientSAM <font></strong></center>"


description_p = """# Interactive Instance Segmentation
                - Point-prompt instruction
                <ol>
                <li> Click on the left image (point input), visualizing the point on the right image </li>
                <li> Click the button of Segment with Point Prompt </li>
                </ol>
                - Box-prompt instruction
                <ol>
                <li> Click on the left image (one point input), visualizing the point on the right image </li>
                <li> Click on the left image (another point input), visualizing the point and the box on the right image</li>
                <li> Click the button of Segment with Box Prompt </li>
                </ol>
              """


css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def clear():
    return None, None, [], []


def get_points_with_draw(image, cond_image, global_points, global_point_label, evt: gr.SelectData):
    if len(global_points) == 0:
        image = copy.deepcopy(cond_image)
    x, y = evt.index[0], evt.index[1]
    label = "Add Mask"
    point_radius, point_color = 15, (
        (255, 255, 0)
        if label == "Add Mask"
        else (
            255,
            0,
            255,
        )
    )
    global_points.append([x, y])
    global_point_label.append(1 if label == "Add Mask" else 0)

    if image is not None:
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            [
                (x - point_radius, y - point_radius),
                (x + point_radius, y + point_radius),
            ],
            fill=point_color,
        )

    return image, global_points, global_point_label


def get_points_with_draw_(image, cond_image, global_points, global_point_label, evt: gr.SelectData):
    if len(global_points) == 0:
        image = copy.deepcopy(cond_image)
    if len(global_points) > 2:
        return image, global_points, global_point_label
    x, y = evt.index[0], evt.index[1]
    label = "Add Mask"
    point_radius, point_color = 15, (
        (255, 255, 0)
        if label == "Add Mask"
        else (
            255,
            0,
            255,
        )
    )
    global_points.append([x, y])
    global_point_label.append(1 if label == "Add Mask" else 0)

    if image is not None:
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            [
                (x - point_radius, y - point_radius),
                (x + point_radius, y + point_radius),
            ],
            fill=point_color,
        )

    if len(global_points) == 2:
        x1, y1 = global_points[0]
        x2, y2 = global_points[1]
        if x1 < x2 and y1 < y2:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        elif x1 < x2 and y1 >= y2:
            draw.rectangle([x1, y2, x2, y1], outline="red", width=5)
            global_points[0][0] = x1
            global_points[0][1] = y2
            global_points[1][0] = x2
            global_points[1][1] = y1
        elif x1 >= x2 and y1 < y2:
            draw.rectangle([x2, y1, x1, y2], outline="red", width=5)
            global_points[0][0] = x2
            global_points[0][1] = y1
            global_points[1][0] = x1
            global_points[1][1] = y2
        elif x1 >= x2 and y1 >= y2:
            draw.rectangle([x2, y2, x1, y1], outline="red", width=5)
            global_points[0][0] = x2
            global_points[0][1] = y2
            global_points[1][0] = x1
            global_points[1][1] = y1

    return image, global_points, global_point_label


def make_demo(segment_with_point_fn: Callable, segment_with_box_fn: Callable):
    cond_img_p = gr.Image(label="Input with Point", value=default_example[0], type="pil")
    cond_img_b = gr.Image(label="Input with Box", value=default_example[0], type="pil")

    segm_img_p = gr.Image(label="Segmented Image with Point-Prompt", interactive=False, type="pil")
    segm_img_b = gr.Image(label="Segmented Image with Box-Prompt", interactive=False, type="pil")

    with gr.Blocks(css=css, title="Efficient SAM") as demo:
        global_points = gr.State([])
        global_point_label = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                # Title
                gr.Markdown(title)

        with gr.Tab("Point mode"):
            # Images
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    cond_img_p.render()

                with gr.Column(scale=1):
                    segm_img_p.render()

            # Submit & Clear
            # ###
            with gr.Row():
                with gr.Column():
                    with gr.Column():
                        segment_btn_p = gr.Button("Segment with Point Prompt", variant="primary")
                        clear_btn_p = gr.Button("Clear", variant="secondary")

                    gr.Markdown("Try some of the examples below ⬇️")
                    gr.Examples(
                        examples=examples,
                        inputs=[cond_img_p],
                        examples_per_page=4,
                    )

                with gr.Column():
                    # Description
                    gr.Markdown(description_p)

        with gr.Tab("Box mode"):
            # Images
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    cond_img_b.render()

                with gr.Column(scale=1):
                    segm_img_b.render()

            # Submit & Clear
            with gr.Row():
                with gr.Column():
                    with gr.Column():
                        segment_btn_b = gr.Button("Segment with Box Prompt", variant="primary")
                        clear_btn_b = gr.Button("Clear", variant="secondary")

                    gr.Markdown("Try some of the examples below ⬇️")
                    gr.Examples(
                        examples=examples,
                        inputs=[cond_img_b],
                        examples_per_page=4,
                    )

                with gr.Column():
                    # Description
                    gr.Markdown(description_p)

        cond_img_p.select(
            get_points_with_draw,
            inputs=[segm_img_p, cond_img_p, global_points, global_point_label],
            outputs=[segm_img_p, global_points, global_point_label],
        )

        cond_img_b.select(
            get_points_with_draw_,
            [segm_img_b, cond_img_b, global_points, global_point_label],
            [segm_img_b, global_points, global_point_label],
        )

        segment_btn_p.click(
            segment_with_point_fn,
            inputs=[cond_img_p, global_points, global_point_label],
            outputs=[segm_img_p, global_points, global_point_label],
        )

        segment_btn_b.click(
            segment_with_box_fn,
            inputs=[cond_img_b, segm_img_b, global_points, global_point_label],
            outputs=[segm_img_b, global_points, global_point_label],
        )

        clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p, global_points, global_point_label])
        clear_btn_b.click(clear, outputs=[cond_img_b, segm_img_b, global_points, global_point_label])
    return demo
