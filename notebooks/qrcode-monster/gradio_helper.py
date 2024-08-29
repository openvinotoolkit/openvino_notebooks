from typing import Callable
import gradio as gr


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Textbox(label="QR Code content"),
            gr.Textbox(label="Text Prompt"),
            gr.Textbox(label="Negative Text Prompt"),
            gr.Number(
                minimum=-1,
                maximum=9999999999,
                step=1,
                value=42,
                label="Seed",
                info="Seed for the random number generator",
            ),
            gr.Slider(
                minimum=0.0,
                maximum=25.0,
                step=0.25,
                value=7,
                label="Guidance Scale",
                info="Controls the amount of guidance the text prompt guides the image generation",
            ),
            gr.Slider(
                minimum=0.5,
                maximum=2.5,
                step=0.01,
                value=1.5,
                label="Controlnet Conditioning Scale",
                info="""Controls the readability/creativity of the QR code.
                High values: The generated QR code will be more readable.
                Low values: The generated QR code will be more creative.
                """,
            ),
            gr.Slider(label="Steps", step=1, value=5, minimum=1, maximum=50),
        ],
        outputs=["image"],
        examples=[
            [
                "Hi OpenVINO",
                "cozy town on snowy mountain slope 8k",
                "blurry unreal occluded",
                42,
                7.7,
                1.4,
                25,
            ],
        ],
    )
    return demo
