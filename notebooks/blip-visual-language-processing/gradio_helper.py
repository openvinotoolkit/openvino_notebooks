from typing import Callable
import gradio as gr


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Image(label="Image"),
            gr.Textbox(
                label="Question",
                info="If this field is empty, an image caption will be generated",
            ),
        ],
        outputs=[gr.Text(label="Answer"), gr.HTML()],
        examples=[["demo.jpg", ""], ["demo.jpg", "how many dogs are in the picture?"]],
        allow_flagging="never",
    )
    return demo
