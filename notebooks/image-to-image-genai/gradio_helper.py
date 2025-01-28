import gradio as gr
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import sys
import math

import openvino_genai as ov_genai


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

examples = [["astronauts.png", "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"], ["dog_drawing.png", "oil painting"]]


def make_demo(pipeline, generator_cls, image_to_tensor):
    def infer(input_image, prompt, negative_prompt, seed, strength, randomize_seed, num_inference_steps, progress=gr.Progress(track_tqdm=True)):
        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)

        generator = generator_cls(seed)
        init_image_tensor = image_to_tensor(input_image)

        pbar = tqdm(total=math.ceil((num_inference_steps + 1) * strength))

        def callback(step, num_steps, latent):
            pbar.update(1)
            sys.stdout.flush()
            return False

        image_tensor = pipeline.generate(
            prompt,
            init_image_tensor,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            strength=strength,
            callback=callback,
        )

        return image_tensor.data[0], seed

    with gr.Blocks() as demo:
        gr.Markdown(
            """
        # Demo Image to Image with OpenVINO GenAI API
        """
        )
        with gr.Row():
            with gr.Column():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                input_image = gr.Image(label="Input image", show_label=False, type="pil")

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

            strength = gr.Slider(
                label="strength",
                minimum=0,
                maximum=1,
                step=0.05,
                value=0.75,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=1,
                maximum=50,
                step=1,
                value=20,
            )

        gr.Examples(examples=examples, inputs=[input_image, prompt])
        gr.on(
            triggers=[run_button.click, prompt.submit, negative_prompt.submit],
            fn=infer,
            inputs=[input_image, prompt, negative_prompt, seed, strength, randomize_seed, num_inference_steps],
            outputs=[result, seed],
        )

    return demo
