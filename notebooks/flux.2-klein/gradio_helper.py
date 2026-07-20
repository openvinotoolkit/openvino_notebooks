import random

import gradio as gr
import numpy as np
import openvino as ov
import openvino_genai as ov_genai

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

EXAMPLE_PROMPTS = [
    "A cat holding a sign that says hello world",
    "Soaking wet capybara taking shelter under a banana leaf in the rainy jungle, close up photo",
    "A kawaii die-cut sticker of a chubby orange cat, featuring big sparkly eyes and a happy smile",
    "Create a vase on a table in living room, the color of the vase is a gradient of color",
]


def make_demo(ov_pipe, model_dir, device="CPU"):
    """Build Gradio demo for FLUX.2 Klein with OpenVINO GenAI.

    Args:
        ov_pipe: openvino_genai.Text2ImagePipeline instance.
        model_dir: Path to the exported OpenVINO model (used to build the editing pipeline).
        device: Inference device for the editing pipeline.
    """
    from PIL import Image

    # Image editing pipeline is created lazily from the exported model on first use.
    edit_state = {"pipe": None}

    def get_edit_pipe():
        if edit_state["pipe"] is None:
            edit_state["pipe"] = ov_genai.Image2ImagePipeline(str(model_dir), device)
        return edit_state["pipe"]

    def generate(
        prompt,
        input_images,
        seed=0,
        randomize_seed=True,
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=1.0,
        progress=gr.Progress(track_tqdm=True),
    ):
        if not prompt.strip():
            raise gr.Error("Please enter a prompt.")

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)  # nosec B311 - UI seed, not security

        generator = ov_genai.TorchGenerator(int(seed))

        # Collect reference image (if provided) for image editing
        ref_image = None
        if input_images is not None and len(input_images) > 0:
            item = input_images[0]
            if isinstance(item, tuple):
                ref_image = item[0]
            elif isinstance(item, Image.Image):
                ref_image = item
            else:
                ref_image = Image.open(item)

        if ref_image is not None:
            ref_tensor = ov.Tensor(np.array(ref_image.convert("RGB"))[None])
            image_tensor = get_edit_pipe().generate(
                prompt,
                ref_tensor,
                strength=0.9,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        else:
            image_tensor = ov_pipe.generate(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        image = Image.fromarray(image_tensor.data[0])

        return image, seed

    with gr.Blocks(title="FLUX.2 Klein — OpenVINO") as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown("""# FLUX.2 [Klein] — OpenVINO
FLUX.2 [klein] is a fast, unified image generation and editing model accelerated by OpenVINO.
            """)

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        prompt = gr.Text(
                            label="Prompt",
                            show_label=False,
                            max_lines=2,
                            placeholder="Enter your prompt",
                            container=False,
                            scale=3,
                        )
                        run_button = gr.Button("Run", scale=1)

                    with gr.Accordion("Input image(s) (optional — for editing)", open=False):
                        input_images = gr.Gallery(
                            label="Input Image(s)",
                            type="pil",
                            columns=3,
                            rows=1,
                        )

                    with gr.Accordion("Advanced Settings", open=False):
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=0,
                        )
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                        with gr.Row():
                            width = gr.Slider(
                                label="Width",
                                minimum=256,
                                maximum=MAX_IMAGE_SIZE,
                                step=16,
                                value=1024,
                            )
                            height = gr.Slider(
                                label="Height",
                                minimum=256,
                                maximum=MAX_IMAGE_SIZE,
                                step=16,
                                value=1024,
                            )

                        with gr.Row():
                            num_inference_steps = gr.Slider(
                                label="Number of inference steps",
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=4,
                            )
                            guidance_scale = gr.Slider(
                                label="Guidance scale",
                                minimum=0.0,
                                maximum=10.0,
                                step=0.1,
                                value=1.0,
                            )

                with gr.Column():
                    result = gr.Image(label="Result", show_label=False)

            gr.Examples(
                examples=EXAMPLE_PROMPTS,
                inputs=[prompt],
            )

        gr.on(
            triggers=[run_button.click, prompt.submit],
            fn=generate,
            inputs=[prompt, input_images, seed, randomize_seed, width, height, num_inference_steps, guidance_scale],
            outputs=[result, seed],
        )

    return demo
