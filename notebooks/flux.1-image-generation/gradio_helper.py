import gradio as gr
import numpy as np
import openvino_genai as ov_genai

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""


def make_demo(ov_pipe, model_name):
    is_guidance_scale_supported = model_name != "FLUX.1-schnell"

    def infer(prompt, seed=42, randomize_seed=False, width=1024, height=1024, num_inference_steps=4, guidance_scale=0, progress=gr.Progress(track_tqdm=True)):
        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)
        generator = ov_genai.TorchGenerator(seed)
        image = ov_pipe.generate(
            prompt=prompt, width=width, height=height, num_inference_steps=num_inference_steps, generator=generator, guidance_scale=guidance_scale
        ).data[0]
        return image, seed

    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown(f"""# FLUX.1 OpenVINO demo""")

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
                        step=32,
                        value=512,
                    )

                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=512,
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=15,
                        step=0.1,
                        value=3.5 if is_guidance_scale_supported else 0.0,
                        visible=is_guidance_scale_supported,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=4,
                    )

            gr.Examples(
                examples=examples,
                inputs=[prompt],
            )

        gr.on(
            triggers=[run_button.click, prompt.submit],
            fn=infer,
            inputs=[prompt, seed, randomize_seed, width, height, num_inference_steps, guidance_scale],
            outputs=[result, seed],
        )
    return demo
