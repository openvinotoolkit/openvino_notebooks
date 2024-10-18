from typing import Callable
from pathlib import Path
import gradio as gr


def make_demo(fn: Callable, quantized: bool, sample_path: Path):
    demo = gr.Interface(
        description=f"""
                    <div style="text-align: center; max-width: 700px; margin: 0 auto;">
                    <div
                        style="
                        display: grid; align-items: center; gap: 0.8rem; font-size: 1.75rem;
                        "
                    >
                        <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                            OpenVINO Generate API Whisper demo {'with quantized model.' if quantized else ''}
                        </h1>
                        <div style="font-size: 12px;">
                            If you use video more then 30 sec, please, note, max_length will be increased. You also could be update it useing generation_config.
                        </div>
                    </div>
                    </div>
                """,
        fn=fn,
        inputs=[
            gr.Video(label="Video"),
            gr.Radio(["Transcribe", "Translate"], value="Transcribe"),
            gr.Checkbox(
                value=quantized,
                visible=quantized,
                label="Use INT8",
            ),
        ],
        outputs="video",
        examples=[[sample_path, "Transcribe"]],
        allow_flagging="never",
    )

    return demo
