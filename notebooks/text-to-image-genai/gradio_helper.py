import gradio as gr
import numpy as np

import openvino_genai


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 832

examples = [
    "Cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting",
    "Curly-haired unicorn in the forest, anime, line",
    "Snime, masterpiece, high quality, a green snowman with a happy smiling face in the snows",
    "A panda reading a book in a lush forest.",
    "Pirate ship sailing on a sea with the milky way galaxy in the sky and purple glow lights",
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
"""


def make_demo(pipeline, generator_cls, adapter_config):
    def infer(prompt, negative_prompt, seed, randomize_seed, width, height, num_inference_steps, use_lora, progress=gr.Progress(track_tqdm=True)):
        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)

        generator = generator_cls(seed)

        image_tensor = pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            adapters=adapter_config if use_lora else openvino_genai.AdapterConfig(),
        )

        return image_tensor.data[0]

    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown(
                """
            # Demo Text to Image with OpenVINO with Generate API
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
            use_lora = gr.Checkbox(label="Use LoRA", value=False)

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
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=20,
                    )

            gr.Examples(examples=examples, inputs=[prompt])
        gr.on(
            triggers=[run_button.click, prompt.submit, negative_prompt.submit],
            fn=infer,
            inputs=[prompt, negative_prompt, seed, randomize_seed, width, height, num_inference_steps, use_lora],
            outputs=[result],
        )

    return demo
