import io
import copy
import requests
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from PIL import Image, ImageDraw


DESCRIPTION = "# Florence-2 OpenVINO Demo"

colormap = [
    "blue",
    "orange",
    "green",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "red",
    "lime",
    "indigo",
    "violet",
    "aqua",
    "magenta",
    "coral",
    "gold",
    "tan",
    "skyblue",
]


example_images = [
    ("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true", "car.jpg"),
    ("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11", "cat.png"),
    ("https://github.com/user-attachments/assets/8c9ae017-7837-4abc-ae92-c1054c9ec350", "hand-written.png"),
]


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)


def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, label in zip(data["bboxes"], data["labels"]):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        plt.text(x1, y1, label, color="white", fontsize=8, bbox=dict(facecolor="red", alpha=0.5))
    ax.axis("off")
    return fig


def draw_polygons(image, prediction, fill_mask=False):
    draw = ImageDraw.Draw(image)
    scale = 1
    for polygons, label in zip(prediction["polygons"], prediction["labels"]):
        color_id = np.random.choice(len(colormap))
        color = colormap[color_id]
        if fill_mask:
            fill_color_id = np.random.choice(len(colormap))
            fill_color = colormap[fill_color_id]
        else:
            fill_color = None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print("Invalid polygon:", _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return image


def convert_to_od_format(data):
    bboxes = data.get("bboxes", [])
    labels = data.get("bboxes_labels", [])
    od_results = {"bboxes": bboxes, "labels": labels}
    return od_results


def draw_ocr_bboxes(image, prediction):
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction["quad_boxes"], prediction["labels"]
    for box, label in zip(bboxes, labels):
        color_id = np.random.choice(len(colormap))
        color = colormap[color_id]
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0] + 8, new_box[1] + 2), "{}".format(label), align="right", fill=color)
    return image


css = """
  #output {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""


single_task_list = [
    "Caption",
    "Detailed Caption",
    "More Detailed Caption",
    "Object Detection",
    "Dense Region Caption",
    "Region Proposal",
    "Caption to Phrase Grounding",
    "Referring Expression Segmentation",
    "Region to Segmentation",
    "Open Vocabulary Detection",
    "Region to Category",
    "Region to Description",
    "OCR",
    "OCR with Region",
]

cascased_task_list = ["Caption + Grounding", "Detailed Caption + Grounding", "More Detailed Caption + Grounding"]


def update_task_dropdown(choice):
    if choice == "Cascased task":
        return gr.Dropdown(choices=cascased_task_list, value="Caption + Grounding")
    else:
        return gr.Dropdown(choices=single_task_list, value="Caption")


def make_demo(model, processor):
    for url, filename in example_images:
        if not Path(filename).exists():
            image = Image.open(requests.get(url, stream=True).raw)
            image.save(filename)

    def run_example(task_prompt, image, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
        return parsed_answer

    def process_image(image, task_prompt, text_input=None):
        image = Image.fromarray(image)  # Convert NumPy array to PIL Image
        if task_prompt == "Caption":
            task_prompt = "<CAPTION>"
            results = run_example(task_prompt, image)
            return results, None
        elif task_prompt == "Detailed Caption":
            task_prompt = "<DETAILED_CAPTION>"
            results = run_example(task_prompt, image)
            return results, None
        elif task_prompt == "More Detailed Caption":
            task_prompt = "<MORE_DETAILED_CAPTION>"
            results = run_example(task_prompt, image)
            return results, None
        elif task_prompt == "Caption + Grounding":
            task_prompt = "<CAPTION>"
            results = run_example(task_prompt, image)
            text_input = results[task_prompt]
            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            results = run_example(task_prompt, image, text_input)
            results["<CAPTION>"] = text_input
            fig = plot_bbox(image, results["<CAPTION_TO_PHRASE_GROUNDING>"])
            return results, fig_to_pil(fig)
        elif task_prompt == "Detailed Caption + Grounding":
            task_prompt = "<DETAILED_CAPTION>"
            results = run_example(task_prompt, image)
            text_input = results[task_prompt]
            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            results = run_example(task_prompt, image, text_input)
            results["<DETAILED_CAPTION>"] = text_input
            fig = plot_bbox(image, results["<CAPTION_TO_PHRASE_GROUNDING>"])
            return results, fig_to_pil(fig)
        elif task_prompt == "More Detailed Caption + Grounding":
            task_prompt = "<MORE_DETAILED_CAPTION>"
            results = run_example(task_prompt, image)
            text_input = results[task_prompt]
            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            results = run_example(task_prompt, image, text_input)
            results["<MORE_DETAILED_CAPTION>"] = text_input
            fig = plot_bbox(image, results["<CAPTION_TO_PHRASE_GROUNDING>"])
            return results, fig_to_pil(fig)
        elif task_prompt == "Object Detection":
            task_prompt = "<OD>"
            results = run_example(task_prompt, image)
            fig = plot_bbox(image, results["<OD>"])
            return results, fig_to_pil(fig)
        elif task_prompt == "Dense Region Caption":
            task_prompt = "<DENSE_REGION_CAPTION>"
            results = run_example(task_prompt, image)
            fig = plot_bbox(image, results["<DENSE_REGION_CAPTION>"])
            return results, fig_to_pil(fig)
        elif task_prompt == "Region Proposal":
            task_prompt = "<REGION_PROPOSAL>"
            results = run_example(task_prompt, image)
            fig = plot_bbox(image, results["<REGION_PROPOSAL>"])
            return results, fig_to_pil(fig)
        elif task_prompt == "Caption to Phrase Grounding":
            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            results = run_example(task_prompt, image, text_input)
            fig = plot_bbox(image, results["<CAPTION_TO_PHRASE_GROUNDING>"])
            return results, fig_to_pil(fig)
        elif task_prompt == "Referring Expression Segmentation":
            task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
            results = run_example(task_prompt, image, text_input)
            output_image = copy.deepcopy(image)
            output_image = draw_polygons(output_image, results["<REFERRING_EXPRESSION_SEGMENTATION>"], fill_mask=True)
            return results, output_image
        elif task_prompt == "Region to Segmentation":
            task_prompt = "<REGION_TO_SEGMENTATION>"
            results = run_example(task_prompt, image, text_input)
            output_image = copy.deepcopy(image)
            output_image = draw_polygons(output_image, results["<REGION_TO_SEGMENTATION>"], fill_mask=True)
            return results, output_image
        elif task_prompt == "Open Vocabulary Detection":
            task_prompt = "<OPEN_VOCABULARY_DETECTION>"
            results = run_example(task_prompt, image, text_input)
            bbox_results = convert_to_od_format(results["<OPEN_VOCABULARY_DETECTION>"])
            fig = plot_bbox(image, bbox_results)
            return results, fig_to_pil(fig)
        elif task_prompt == "Region to Category":
            task_prompt = "<REGION_TO_CATEGORY>"
            results = run_example(task_prompt, image, text_input)
            return results, None
        elif task_prompt == "Region to Description":
            task_prompt = "<REGION_TO_DESCRIPTION>"
            results = run_example(task_prompt, image, text_input)
            return results, None
        elif task_prompt == "OCR":
            task_prompt = "<OCR>"
            results = run_example(task_prompt, image)
            return results, None
        elif task_prompt == "OCR with Region":
            task_prompt = "<OCR_WITH_REGION>"
            results = run_example(task_prompt, image)
            output_image = copy.deepcopy(image)
            output_image = draw_ocr_bboxes(output_image, results["<OCR_WITH_REGION>"])
            return results, output_image
        else:
            return "", None

    with gr.Blocks(css=css) as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tab(label="Florence-2 Image Captioning"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(label="Input Picture")
                    task_type = gr.Radio(choices=["Single task", "Cascased task"], label="Task type selector", value="Single task")
                    task_prompt = gr.Dropdown(choices=single_task_list, label="Task Prompt", value="Caption")
                    task_type.change(fn=update_task_dropdown, inputs=task_type, outputs=task_prompt)
                    text_input = gr.Textbox(label="Text Input (optional)")
                    submit_btn = gr.Button(value="Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output Text")
                    output_img = gr.Image(label="Output Image")

            gr.Examples(
                examples=[["car.jpg", "Region to Segmentation"], ["hand-written.png", "OCR with Region"], ["cat.png", "Detailed Caption"]],
                inputs=[input_img, task_prompt],
                label="Try examples",
            )

            submit_btn.click(process_image, [input_img, task_prompt, text_input], [output_text, output_img])

    return demo
