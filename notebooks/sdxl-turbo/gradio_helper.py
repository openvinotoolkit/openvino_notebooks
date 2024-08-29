from typing import Callable
import gradio as gr


def make_demo(fn: Callable, quantized: bool):
    with gr.Blocks() as demo:
        with gr.Column():
            positive_input = gr.Textbox(label="Text prompt")
            with gr.Row():
                seed_input = gr.Number(precision=0, label="Seed", value=42, minimum=0)
                steps_input = gr.Slider(label="Steps", value=1, minimum=1, maximum=4, step=1)
                height_input = gr.Slider(label="Height", value=512, minimum=256, maximum=1024, step=32)
                width_input = gr.Slider(label="Width", value=512, minimum=256, maximum=1024, step=32)
                btn = gr.Button()
            output_image = gr.Image(
                label=("Result (Quantized)" if quantized else "Result (Original)"),
                type="pil",
                width=512,
            )
            btn.click(
                fn=fn,
                inputs=[positive_input, seed_input, steps_input, height_input, width_input],
                outputs=output_image,
            )
            gr.Examples(
                [
                    ["cute cat", 999],
                    [
                        "underwater world coral reef, colorful jellyfish, 35mm, cinematic lighting, shallow depth of field,  ultra quality, masterpiece, realistic",
                        89,
                    ],
                    [
                        "a photo realistic happy white poodle dog ​​playing in the grass, extremely detailed, high res, 8k, masterpiece, dynamic angle",
                        1569,
                    ],
                    [
                        "Astronaut on Mars watching sunset, best quality, cinematic effects,",
                        65245,
                    ],
                    [
                        "Black and white street photography of a rainy night in New York, reflections on wet pavement",
                        48199,
                    ],
                ],
                [positive_input, seed_input],
            )
    return demo
