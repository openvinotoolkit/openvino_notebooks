from typing import Callable
import gradio as gr


def make_demo(fn: Callable):
    audio1 = gr.Audio(label="Source Audio", type="filepath")
    audio2 = gr.Audio(label="Reference Audio", type="filepath")
    outputs = gr.Audio(label="Output Audio", type="filepath")
    examples = [["p225_001.wav", "p226_002.wav"]]

    title = "FreeVC with Gradio"
    description = 'Gradio Demo for FreeVC and OpenVINO™. Upload a source audio and a reference audio, then click the "Submit" button to inference.'

    demo = gr.Interface(
        fn=fn,
        inputs=[audio1, audio2],
        outputs=outputs,
        title=title,
        description=description,
        examples=examples,
    )
    return demo
