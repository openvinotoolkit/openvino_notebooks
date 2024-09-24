from typing import Callable
import gradio as gr


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Audio(label="Source Audio", type="filepath"),
            gr.Slider(-100, 100, value=0, label="Pitch shift", step=1),
            gr.Slider(
                -80,
                -20,
                value=-30,
                label="Slice db",
                step=10,
                info="The default is -30, noisy audio can be -30, dry sound can be -50 to preserve breathing.",
            ),
            gr.Slider(
                0,
                1,
                value=0.4,
                label="Noise scale",
                step=0.1,
                info="Noise level will affect pronunciation and sound quality, which is more metaphysical",
            ),
        ],
        outputs=gr.Audio(label="Output Audio", type="numpy"),
        title="SoftVC VITS Singing Voice Conversion with Gradio",
        description="Gradio Demo for SoftVC VITS Singing Voice Conversion and OpenVINO™. Upload a source audio, then click the 'Submit' button to inference.",
        examples=[["raw/000.wav", 0, -30, 0.4, False]],
    )
    return demo
