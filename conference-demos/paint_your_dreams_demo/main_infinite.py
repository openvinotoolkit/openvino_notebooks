import argparse
import time

import gradio as gr
import numpy as np
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from optimum.intel.openvino import OVLatentConsistencyModelPipeline

stop_generating: bool = False
ov_pipeline: OVLatentConsistencyModelPipeline | None = None
safety_checker: StableDiffusionSafetyChecker | None = None


def load_pipeline(model_name, device):
    global ov_pipeline, safety_checker

    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    ov_pipeline = OVLatentConsistencyModelPipeline.from_pretrained(model_name, compile=False, safety_checker=safety_checker)

    ov_pipeline.to(device)
    ov_pipeline.compile()


def stop():
    global stop_generating
    stop_generating = True


def generate_images(prompt: str, size: int, guidance_scale: float, num_inference_steps: int):
    global stop_generating

    stop_generating = False
    while True:
        if stop_generating:
            break

        start_time = time.time()
        result = ov_pipeline(prompt=prompt, num_inference_steps=num_inference_steps, width=size, height=size, guidance_scale=guidance_scale).images
        end_time = time.time()

        result, nsfw = safety_checker(ov_pipeline.feature_extractor(result, return_tensors="pt").pixel_values, np.array(result))

        processing_time = end_time - start_time
        yield result[0], round(processing_time, 5)


def build_ui():
    examples = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour,"
        "style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
    ]

    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    value=examples[0],
                    label="Prompt",
                    max_lines=1,
                    placeholder="Enter your prompt here",
                )
            with gr.Row():
                gr.Column(scale=1)
                with gr.Column(scale=6):
                    result = gr.Image(label="Generated image", elem_id="output_image", format="png")
                    result_time_label = gr.Text("", label="Processing Time", type="text")
                    with gr.Row():
                        run_button = gr.Button("Start generation")
                        stop_button = gr.Button("Stop generation", interactive=False)
                gr.Column(scale=1)
            with gr.Accordion("Advanced options", open=False):
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance scale for base",
                        minimum=2,
                        maximum=14,
                        step=0.1,
                        value=8.0,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps for base",
                        minimum=1,
                        maximum=32,
                        step=1,
                        value=10,
                    )

                size = gr.Slider(
                    label="Image size",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=512
                )

        gr.Examples(
            examples=examples,
            inputs=prompt,
            outputs=result,
            cache_examples=False,
        )
        gr.on(
            triggers=[
                prompt.submit,
                run_button.click,
            ],
            fn=generate_images,
            inputs=[
                prompt, size, guidance_scale, num_inference_steps
            ],
            outputs=[
                result, result_time_label
            ],
        )
        run_button.click(lambda: gr.Button(interactive=False), outputs=run_button) \
            .then(lambda: gr.Button(interactive=True), outputs=stop_button)
        stop_button.click(lambda: gr.Button(interactive=False), outputs=stop_button) \
            .then(lambda: gr.Button(interactive=True), outputs=run_button) \
            .then(stop)

    return demo


def run_endless_lcm(model_name, device):
    load_pipeline(model_name, device)

    demo = build_ui()
    demo.launch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="GPU", type=str, help="Device to run inference on")
    parser.add_argument("--model_name", type=str, default="OpenVINO/LCM_Dreamshaper_v7-int8-ov",
                        help="Pose estimation model to be used")

    args = parser.parse_args()
    run_endless_lcm(args.model_name, args.device)
