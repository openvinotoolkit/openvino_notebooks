import argparse
import os
from datetime import datetime

import gradio as gr
import numpy as np
import torch
from PIL import Image

from model.cloth_masker import vis_mask
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


HEADER = """
<h1 style="text-align: center;"> 🐈 CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models </h1>
"""


def make_demo(pipeline, mask_processor, automasker, output_dir):
    def submit_function(person_image, cloth_image, cloth_type, num_inference_steps, guidance_scale, seed, show_type):
        width = 768
        height = 1024
        person_image, mask = person_image["background"], person_image["layers"][0]
        mask = Image.open(mask).convert("L")
        if len(np.unique(np.array(mask))) == 1:
            mask = None
        else:
            mask = np.array(mask)
            mask[mask > 0] = 255
            mask = Image.fromarray(mask)

        tmp_folder = output_dir
        date_str = datetime.now().strftime("%Y%m%d%H%M%S")
        result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
        if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
            os.makedirs(os.path.join(tmp_folder, date_str[:8]))

        generator = None
        if seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        person_image = Image.open(person_image).convert("RGB")
        cloth_image = Image.open(cloth_image).convert("RGB")
        person_image = resize_and_crop(person_image, (width, height))
        cloth_image = resize_and_padding(cloth_image, (width, height))

        # Process mask
        if mask is not None:
            mask = resize_and_crop(mask, (width, height))
        else:
            mask = automasker(person_image, cloth_type)["mask"]
        mask = mask_processor.blur(mask, blur_factor=9)

        # Inference
        result_image = pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )[0]

        # Post-process
        masked_person = vis_mask(person_image, mask)
        save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
        save_result_image.save(result_save_path)
        if show_type == "result only":
            return result_image
        else:
            width, height = person_image.size
            if show_type == "input & result":
                condition_width = width // 2
                conditions = image_grid([person_image, cloth_image], 2, 1)
            else:
                condition_width = width // 3
                conditions = image_grid([person_image, masked_person, cloth_image], 3, 1)
            conditions = conditions.resize((condition_width, height), Image.NEAREST)
            new_result_image = Image.new("RGB", (width + condition_width + 5, height))
            new_result_image.paste(conditions, (0, 0))
            new_result_image.paste(result_image, (condition_width + 5, 0))
        return new_result_image

    with gr.Blocks(title="CatVTON") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                with gr.Row():
                    person_image = gr.ImageEditor(interactive=True, label="Person Image", type="filepath")

                with gr.Row():
                    with gr.Column(scale=1, min_width=230):
                        cloth_image = gr.Image(interactive=True, label="Condition Image", type="filepath")
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">Two ways to provide Mask:<br>1. Upload the person image and use the `🖌️` above to draw the Mask (higher priority)<br>2. Select the `Try-On Cloth Type` to generate automatically </span>'
                        )
                        cloth_type = gr.Radio(
                            label="Try-On Cloth Type",
                            choices=["upper", "lower", "overall"],
                            value="upper",
                        )

                submit = gr.Button("Submit")

                gr.Markdown(
                    '<span style="color: #808080; font-size: small;">Advanced options can adjust details:<br>1. `Inference Step` may enhance details;<br>2. `CFG` is highly correlated with saturation;<br>3. `Random seed` may improve pseudo-shadow.</span>'
                )
                with gr.Accordion("Advanced Options", open=False):
                    num_inference_steps = gr.Slider(label="Inference Step", minimum=10, maximum=100, step=5, value=50)
                    # Guidence Scale
                    guidance_scale = gr.Slider(label="CFG Strenth", minimum=0.0, maximum=7.5, step=0.5, value=2.5)
                    # Random Seed
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=10000, step=1, value=42)
                    show_type = gr.Radio(
                        label="Show Type",
                        choices=["result only", "input & result", "input & mask & result"],
                        value="input & mask & result",
                    )

            with gr.Column(scale=2, min_width=500):
                result_image = gr.Image(interactive=False, label="Result")
                with gr.Row():
                    # Photo Examples
                    root_path = "CatVTON/resource/demo/example"
                    with gr.Column():
                        men_exm = gr.Examples(
                            examples=[os.path.join(root_path, "person", "men", _) for _ in os.listdir(os.path.join(root_path, "person", "men"))],
                            examples_per_page=4,
                            inputs=person_image,
                            label="Person Examples ①",
                        )
                        women_exm = gr.Examples(
                            examples=[os.path.join(root_path, "person", "women", _) for _ in os.listdir(os.path.join(root_path, "person", "women"))],
                            examples_per_page=4,
                            inputs=person_image,
                            label="Person Examples ②",
                        )
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">*Person examples come from the demos of <a href="https://huggingface.co/spaces/levihsu/OOTDiffusion">OOTDiffusion</a> and <a href="https://www.outfitanyone.org">OutfitAnyone</a>. </span>'
                        )
                    with gr.Column():
                        condition_upper_exm = gr.Examples(
                            examples=[os.path.join(root_path, "condition", "upper", _) for _ in os.listdir(os.path.join(root_path, "condition", "upper"))],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Upper Examples",
                        )
                        condition_overall_exm = gr.Examples(
                            examples=[os.path.join(root_path, "condition", "overall", _) for _ in os.listdir(os.path.join(root_path, "condition", "overall"))],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Overall Examples",
                        )
                        condition_person_exm = gr.Examples(
                            examples=[os.path.join(root_path, "condition", "person", _) for _ in os.listdir(os.path.join(root_path, "condition", "person"))],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Reference Person Examples",
                        )
                        gr.Markdown('<span style="color: #808080; font-size: small;">*Condition examples come from the Internet. </span>')

            submit.click(
                submit_function,
                [
                    person_image,
                    cloth_image,
                    cloth_type,
                    num_inference_steps,
                    guidance_scale,
                    seed,
                    show_type,
                ],
                result_image,
            )

    return demo
