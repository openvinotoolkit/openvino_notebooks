import gradio as gr
import numpy as np

import openvino_genai as ov_genai


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048


def make_demo(pipeline, generator_cls, adapters, adapters_meta):

    adapters_selection = {"none": {"prompt": "<subject>"}}
    for idx, adapter in enumerate(adapters):
        adapter_name = adapters_meta[idx]["name"]
        adapters_selection[adapter_name] = adapters_meta[idx]
        adapters_selection[adapter_name]["adapter"] = adapter

    def infer(prompt, seed, randomize_seed, width, height, lora_id, progress=gr.Progress(track_tqdm=True)):
        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)

        generator = generator_cls(seed)
        adapter_config = ov_genai.AdapterConfig()
        if lora_id != "none":
            adapter_info = adapters_selection[lora_id]
            adapter = adapter_info["adapter"]
            prompt_template = adapter_info.get("prompt", "<subject>")
            alpha = adapter_info.get("weight", 1.0)
            adapter_config.add(adapter, alpha)
            prompt = prompt_template.replace("<subject>", prompt)

        image_tensor = pipeline.generate(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0,
            width=width,
            height=height,
            generator=generator,
            adapters=adapter_config,
        )

        return image_tensor.data[0], seed

    with gr.Blocks() as demo:
        with gr.Column():
            gr.Markdown(
                """
            # Image Generation with LoRA and OpenVINO GenAI
            1. Provide input generation prompt into prompt window
            2. Select one of the predefined adapters (use none for a generation without LoRA)
            3. Click 'Generate' button for start image generation
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
                lora_id = gr.Dropdown(label="LoRA", choices=list(adapters_selection.keys()), value="none")

                run_button = gr.Button("Generate", scale=0)

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
                        value=1024,
                    )

                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )

        gr.on(
            triggers=[run_button.click, prompt.submit],
            fn=infer,
            inputs=[prompt, seed, randomize_seed, width, height, lora_id],
            outputs=[result, seed],
        )

    return demo
