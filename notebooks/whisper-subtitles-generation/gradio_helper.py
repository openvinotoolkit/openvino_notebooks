from typing import Callable
import gradio as gr


def make_demo(fn: Callable, quantized: bool):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Textbox(label="YouTube URL"),
            gr.Radio(["Transcribe", "Translate"], value="Transcribe"),
            gr.Checkbox(
                value=quantized,
                visible=quantized,
                label="Use INT8",
            ),
        ],
        outputs="video",
        examples=[["https://youtu.be/kgL5LBM-hFI", "Transcribe"]],
        allow_flagging="never",
    )

    return demo
