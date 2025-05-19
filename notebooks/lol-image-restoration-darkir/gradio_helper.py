from pathlib import Path
import gradio as gr

examples = [[img] for img in Path("./DarkIR/assets/qualis/inputs").glob("*.png")]


def make_demo(fn):
    demo = gr.Interface(
        fn=fn, inputs=[gr.Image(type="pil", label="input")], outputs=[gr.Image(type="pil", label="output")], title="Low-Light-Deblurring", examples=examples
    )
    return demo
