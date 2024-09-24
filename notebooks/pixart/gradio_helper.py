from typing import Callable
import gradio as gr
import numpy as np

MAX_SEED = np.iinfo(np.int32).max

examples = [
    ["A small cactus with a happy face in the Sahara desert.", 42],
    ["an astronaut sitting in a diner, eating fries, cinematic, analog film", 42],
    [
        "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
        0,
    ],
    ["professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.", 0],
]


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Textbox(label="Caption"),
            gr.Slider(0, MAX_SEED, label="Seed"),
            gr.Textbox(label="Negative prompt"),
            gr.Slider(2, 20, step=1, label="Number of inference steps", value=4),
        ],
        outputs="image",
        examples=examples,
        allow_flagging="never",
    )
    return demo
