from pathlib import Path
from typing import Callable
import gradio as gr
import numpy as np
from diffusers.utils import load_image
from style_template import styles

STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"

MAX_SEED = np.iinfo(np.int32).max

title = r"""
<h1 align="center">InstantID: Zero-shot Identity-Preserving Generation</h1>
"""

description = r"""

    How to use:<br>
    1. Upload an image with a face. For images with multiple faces, we will only detect the largest face. Ensure the face is not too small and is clearly visible without significant obstructions or blurring.
    2. (Optional) You can upload another image as a reference for the face pose. If you don't, we will use the first detected face image to extract facial landmarks. If you use a cropped face at step 1, it is recommended to upload it to define a new face pose.
    3. Enter a text prompt, as done in normal text-to-image models.
    4. Click the <b>Submit</b> button to begin customization.
    5. Share your customized photo with your friends and enjoy! 😊
    """

css = """
    .gradio-container {width: 85% !important}
    """


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)
    return seed


def get_examples():
    example_image_urls = [
        "https://huggingface.co/datasets/EnD-Diffusers/AI_Faces/resolve/main/00002-3104853212.png",
        "https://huggingface.co/datasets/EnD-Diffusers/AI_Faces/resolve/main/images%207/00171-2728008415.png",
        "https://huggingface.co/datasets/EnD-Diffusers/AI_Faces/resolve/main/00003-3962843561.png",
        "https://huggingface.co/datasets/EnD-Diffusers/AI_Faces/resolve/main/00005-3104853215.png",
        "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ai_face2.png",
    ]

    examples_dir = Path("examples")

    if not examples_dir.exists():
        examples_dir.mkdir()
        for img_id, img_url in enumerate(example_image_urls):
            out_path = examples_dir / f"face_{img_id}.png"
            if not out_path.exists():
                load_image(img_url).save(examples_dir / f"face_{img_id}.png")

    return [
        [examples_dir / "face_0.png", "A woman in red dress", "Film Noir", ""],
        [examples_dir / "face_1.png", "photo of a business lady", "Vibrant Color", ""],
        [examples_dir / "face_2.png", "famous rock star poster", "(No style)", ""],
        [examples_dir / "face_3.png", "a person", "Neon", ""],
        [examples_dir / "face_4.png", "a girl", "Snow", ""],
    ]


def make_demo(fn: Callable):
    with gr.Blocks(css=css) as demo:
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                # upload face image
                face_file = gr.Image(label="Upload a photo of your face", type="pil")

                # optional: upload a reference pose image
                pose_file = gr.Image(label="Upload a reference pose image (optional)", type="pil")

                # prompt
                prompt = gr.Textbox(
                    label="Prompt",
                    info="Give simple prompt is enough to achieve good face fidelity",
                    placeholder="A photo of a person",
                    value="",
                )

                submit = gr.Button("Submit", variant="primary")
                style = gr.Dropdown(label="Style template", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)

                # strength
                identitynet_strength_ratio = gr.Slider(
                    label="IdentityNet strength (for fidelity)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )

                with gr.Accordion(open=False, label="Advanced Options"):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="low quality",
                        value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
                    )
                    num_steps = gr.Slider(
                        label="Number of sample steps",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=4,
                    )
                    guidance_scale = gr.Slider(label="Guidance scale", minimum=0.1, maximum=10.0, step=0.1, value=0)
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=42,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                gr.Examples(
                    examples=get_examples(),
                    inputs=[face_file, prompt, style, negative_prompt],
                )

            with gr.Column():
                gallery = gr.Image(label="Generated Image")

        submit.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            api_name=False,
        ).then(
            fn=fn,
            inputs=[
                face_file,
                pose_file,
                prompt,
                negative_prompt,
                style,
                num_steps,
                identitynet_strength_ratio,
                guidance_scale,
                seed,
            ],
            outputs=[gallery],
        )

    return demo
