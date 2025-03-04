import torch
import gradio as gr
import numpy as np
import sys
import openvino_genai as ov_genai
from tqdm.auto import tqdm
from PIL import Image

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344

examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
    "A panda reading a book in a lush forest.",
    "A 3d render of a futuristic city with a giant robot in the middle full of neon lights, pink and blue colors",
    'a wizard kitten holding a sign saying "openvino" with a magic wand.',
    "photo of a huge red cat with green eyes sitting on a cloud in the sky, looking at the camera",
    "Pirate ship sailing on a sea with the milky way galaxy in the sky and purple glow lights",
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
"""


def make_demo(pipeline, turbo):
    def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, progress=gr.Progress(track_tqdm=True)):
        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)

        generator = ov_genai.TorchGenerator(seed)
        pbar = tqdm(total=num_inference_steps)

        def callback(step, num_steps, latent):
            if num_steps != pbar.total:
                pbar.reset(num_steps)
            pbar.update(1)
            sys.stdout.flush()
            return False

        generate_kwargs = {}

        if guidance_scale > 1:
            generate_kwargs["negative_prompt"] = negative_prompt

        image_tensor = pipeline.generate(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            callback=callback,
            **generate_kwargs
        )
        image = Image.fromarray(image_tensor.data[0])
        return image, seed

    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown(
                """
            # Demo Stable Diffusion 3  with OpenVINO
            """
            )

            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )

                run_button = gr.Button("Run", scale=0)

            result = gr.Image(label="Result", show_label=False)

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt = gr.Text(
                    label="Negative prompt",
                    max_lines=1,
                    placeholder="Enter a negative prompt",
                )

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
                        step=64,
                        value=512,
                    )

                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=64,
                        value=512,
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.0,
                        maximum=10.0 if not turbo else 2,
                        step=0.1,
                        value=5.0 if not turbo else 1.5,
                    )

                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=28 if not turbo else 8,
                    )

            gr.Examples(examples=examples, inputs=[prompt])
        gr.on(
            triggers=[run_button.click, prompt.submit, negative_prompt.submit],
            fn=infer,
            inputs=[prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
            outputs=[result, seed],
        )

    return demo
