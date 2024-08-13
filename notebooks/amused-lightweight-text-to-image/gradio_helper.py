import gradio as gr
import numpy as np
import torch


def make_demo(pipe):
    def generate(prompt, seed, _=gr.Progress(track_tqdm=True)):
        image = pipe(prompt, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        return image

    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(label="Prompt"),
            gr.Slider(0, np.iinfo(np.int32).max, label="Seed", step=1),
        ],
        outputs="image",
        examples=[
            ["happy snowman", 88],
            ["green ghost rider", 0],
            ["kind smiling ghost", 8],
        ],
        allow_flagging="never",
    )
    return demo
