from typing import Callable
import gradio as gr

examples = [["data/one.jpg", "data/two.jpg", 5]]


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Image(label="First image"),
            gr.Image(label="Last image"),
            gr.Slider(
                1,
                8,
                step=1,
                label="Times to interpolate",
                info="""Controls the number of times the frame interpolator is invoked.
            The output will be the interpolation video with (2^value + 1) frames, fps of 30.""",
            ),
        ],
        outputs=gr.Video(),
        examples=examples,
        allow_flagging="never",
    )
    return demo
