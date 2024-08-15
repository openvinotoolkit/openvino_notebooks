import gradio as gr
import numpy as np


def make_demo(generate):
    demo = gr.Interface(
        generate,
        [
            gr.Textbox(label="Prompt"),
            gr.Textbox(label="Negative prompt"),
            gr.Slider(
                0,
                20,
                step=1,
                label="Prior guidance scale",
                info="Higher guidance scale encourages to generate images that are closely "
                "linked to the text `prompt`, usually at the expense of lower image quality. Applies to the prior pipeline",
            ),
            gr.Slider(
                0,
                20,
                step=1,
                label="Decoder guidance scale",
                info="Higher guidance scale encourages to generate images that are closely "
                "linked to the text `prompt`, usually at the expense of lower image quality. Applies to the decoder pipeline",
            ),
            gr.Slider(0, np.iinfo(np.int32).max, label="Seed", step=1),
        ],
        "image",
        examples=[["An image of a shiba inu, donning a spacesuit and helmet", "", 4, 0, 0], ["An armchair in the shape of an avocado", "", 4, 0, 0]],
        allow_flagging="never",
    )

    return demo
