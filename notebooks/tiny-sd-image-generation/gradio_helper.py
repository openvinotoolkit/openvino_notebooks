from pathlib import Path
from typing import Callable
import gradio as gr
import requests

sample_image_name = "tower.jpg"

if not Path(sample_image_name).exists():
    r = requests.get("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/tower.jpg")
    with open(sample_image_name, "wb") as f:
        f.write(r.content)


def make_demo(text_to_text_fn: Callable, image_to_image_fn: Callable):
    with gr.Blocks() as demo:
        with gr.Tab("Text-to-Image generation"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(lines=3, label="Positive prompt")
                    negative_text_input = gr.Textbox(lines=3, label="Negative prompt")
                    seed_input = gr.Slider(0, 10000000, value=751, label="Seed")
                    steps_input = gr.Slider(1, 50, value=20, step=1, label="Steps")
                output = gr.Image(label="Result", type="pil")
            sample_text = "futuristic synthwave city, retro sunset, crystals, spires, volumetric lighting, studio Ghibli style, rendered in unreal engine with clean details"
            sample_text2 = "RAW studio photo of tiny cute happy  cat in a yellow raincoat in the woods, rain, a character portrait, soft lighting, high resolution, photo realistic, extremely detailed"
            negative_sample_text = ""
            negative_sample_text2 = "bad anatomy, blurry, noisy, jpeg artifacts, low quality, geometry, mutation, disgusting. ugly"
            btn = gr.Button()
            btn.click(
                fn=text_to_text_fn,
                inputs=[text_input, negative_text_input, seed_input, steps_input],
                outputs=output,
            )
            gr.Examples(
                [
                    [sample_text, negative_sample_text, 42, 20],
                    [sample_text2, negative_sample_text2, 1561, 25],
                ],
                [text_input, negative_text_input, seed_input, steps_input],
            )
        with gr.Tab("Image-to-Image generation"):
            with gr.Row():
                with gr.Column():
                    i2i_input = gr.Image(label="Image", type="pil")
                    i2i_text_input = gr.Textbox(lines=3, label="Text")
                    i2i_negative_text_input = gr.Textbox(lines=3, label="Negative prompt")
                    i2i_seed_input = gr.Slider(0, 10000000, value=42, label="Seed")
                    i2i_steps_input = gr.Slider(1, 50, value=10, step=1, label="Steps")
                    strength_input = gr.Slider(0, 1, value=0.5, label="Strength")
                i2i_output = gr.Image(label="Result", type="pil")
            i2i_btn = gr.Button()
            sample_i2i_text = "amazing watercolor painting"
            i2i_btn.click(
                fn=image_to_image_fn,
                inputs=[
                    i2i_input,
                    i2i_text_input,
                    i2i_negative_text_input,
                    i2i_seed_input,
                    i2i_steps_input,
                    strength_input,
                ],
                outputs=i2i_output,
            )
            gr.Examples(
                [[sample_image_name, sample_i2i_text, "", 6400023, 40, 0.3]],
                [
                    i2i_input,
                    i2i_text_input,
                    i2i_negative_text_input,
                    i2i_seed_input,
                    i2i_steps_input,
                    strength_input,
                ],
            )
    return demo
