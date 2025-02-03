import gradio as gr
import torch
from diffusers.utils import load_image
from pathlib import Path

example_images = [
    ("woman.png", "https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/woman.png"),
    ("statue.png", "https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/statue.png"),
    ("river.png", "https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/river.png"),
]


def make_demo(ov_pipe):
    for img, url in example_images:
        if not Path(img).exists():
            load_image(url).save(img)

    def generate_from_text(
        positive_prompt,
        negative_prompt,
        ip_adapter_image,
        seed,
        num_steps,
        guidance_scale,
        _=gr.Progress(track_tqdm=True),
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = ov_pipe(
            positive_prompt,
            ip_adapter_image=ip_adapter_image,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
        )
        return result.images[0]

    def generate_from_image(
        img,
        ip_adapter_image,
        positive_prompt,
        negative_prompt,
        seed,
        num_steps,
        guidance_scale,
        strength,
        _=gr.Progress(track_tqdm=True),
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = ov_pipe(
            positive_prompt,
            image=img,
            ip_adapter_image=ip_adapter_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        )
        return result.images[0]

    with gr.Blocks() as demo:
        with gr.Tab("Text-to-Image generation"):
            with gr.Row():
                with gr.Column():
                    ip_adapter_input = gr.Image(label="IP-Adapter Image", type="pil")
                    text_input = gr.Textbox(lines=3, label="Positive prompt")
                    neg_text_input = gr.Textbox(lines=3, label="Negative prompt")
                    with gr.Accordion("Advanced options", open=False):
                        seed_input = gr.Slider(0, 10000000, value=42, label="Seed")
                        steps_input = gr.Slider(1, 12, value=4, step=1, label="Steps")
                        guidance_scale_input = gr.Slider(
                            label="Guidance scale",
                            minimum=0.1,
                            maximum=2,
                            step=0.1,
                            value=0.5,
                        )
                out = gr.Image(label="Result", type="pil")
            btn = gr.Button()
            btn.click(
                generate_from_text,
                [
                    text_input,
                    neg_text_input,
                    ip_adapter_input,
                    seed_input,
                    steps_input,
                    guidance_scale_input,
                ],
                out,
            )
            gr.Examples(
                [
                    [
                        "woman.png",
                        "best quality, high quality",
                        "low resolution",
                    ],
                    [
                        "statue.png",
                        "wearing a hat",
                        "",
                    ],
                ],
                [ip_adapter_input, text_input, neg_text_input],
            )
        with gr.Tab("Image-to-Image generation"):
            with gr.Row():
                with gr.Column():
                    i2i_input = gr.Image(label="Image", type="pil")
                    i2i_ip_adapter_input = gr.Image(label="IP-Adapter Image", type="pil")
                    i2i_text_input = gr.Textbox(lines=3, label="Text")
                    i2i_neg_text_input = gr.Textbox(lines=3, label="Negative prompt")
                    with gr.Accordion("Advanced options", open=False):
                        i2i_seed_input = gr.Slider(0, 10000000, value=42, label="Seed")
                        i2i_steps_input = gr.Slider(1, 12, value=8, step=1, label="Steps")
                        strength_input = gr.Slider(0, 1, value=0.7, label="Strength")
                        i2i_guidance_scale = gr.Slider(
                            label="Guidance scale",
                            minimum=0.1,
                            maximum=2,
                            step=0.1,
                            value=0.5,
                        )
                i2i_out = gr.Image(label="Result")
            i2i_btn = gr.Button()
            i2i_btn.click(
                generate_from_image,
                [
                    i2i_input,
                    i2i_ip_adapter_input,
                    i2i_text_input,
                    i2i_neg_text_input,
                    i2i_seed_input,
                    i2i_steps_input,
                    i2i_guidance_scale,
                    strength_input,
                ],
                i2i_out,
            )
            gr.Examples(
                [
                    [
                        "river.png",
                        "statue.png",
                    ],
                ],
                [i2i_ip_adapter_input, i2i_input],
            )

        return demo
