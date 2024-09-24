from pathlib import Path
from typing import Callable
import gradio as gr
import requests

audio_example_path = Path("example_1.wav")

if not audio_example_path.exists():
    r = requests.get("https://huggingface.co/spaces/distil-whisper/whisper-vs-distil-whisper/resolve/main/assets/example_1.wav")
    with open(audio_example_path, "wb") as f:
        f.write(r.content)


def make_demo(fn: Callable, quantized: bool):
    with gr.Blocks() as demo:
        gr.HTML(
            """
                    <div style="text-align: center; max-width: 700px; margin: 0 auto;">
                    <div
                        style="
                        display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem;
                        "
                    >
                        <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                        OpenVINO Distil-Whisper demo
                        </h1>
                    </div>
                    </div>
                """
        )
        audio = gr.components.Audio(type="filepath", label="Audio input")
        with gr.Row():
            button = gr.Button("Transcribe")
            if quantized:
                button_q = gr.Button("Transcribe quantized")
        with gr.Row():
            infer_time = gr.components.Textbox(label="OpenVINO Distil-Whisper Transcription Time (s)")
            if quantized:
                infer_time_q = gr.components.Textbox(label="OpenVINO Quantized Distil-Whisper Transcription Time (s)")
        with gr.Row():
            transcription = gr.components.Textbox(label="OpenVINO Distil-Whisper Transcription", show_copy_button=True)
            if quantized:
                transcription_q = gr.components.Textbox(
                    label="OpenVINO Quantized Distil-Whisper Transcription",
                    show_copy_button=True,
                )
        button.click(
            fn=fn,
            inputs=audio,
            outputs=[transcription, infer_time],
        )
        if quantized:
            button_q.click(
                fn=fn,
                inputs=[audio, gr.Number(value=1, visible=False)],
                outputs=[transcription_q, infer_time_q],
            )
        gr.Markdown("## Examples")
        gr.Examples(
            [[str(audio_example_path)]],
            audio,
            outputs=[transcription, infer_time],
            fn=fn,
            cache_examples=False,
        )

    return demo
