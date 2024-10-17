from typing import Callable
import gradio as gr


title = "MMS with Gradio"
description = (
    'Gradio Demo for MMS and OpenVINO™. Upload a source audio, then click the "Submit" button to detect a language ID and a transcription. '
    "Make sure that the audio data is sampled to 16000 kHz. If this language has not been used before, it may take some time to prepare the ASR model."
    "\n"
    "> Note: In order to run quantized model to transcribe some language, first the quantized model for that specific language must be prepared."
)


def make_demo(fn: Callable, quantized: bool):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(f"# {title}")
        with gr.Row():
            gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                audio = gr.Audio(label="Source Audio", type="filepath")
                run_button = gr.Button(value="Run FP32")
                # if quantized:
                #     run_quantized_button = gr.Button(value="Run INT8")
            with gr.Column():
                detected_language = gr.Textbox(label="Detected language ID")
                transcription = gr.Textbox(label="Transcription")
                identification_time = gr.Textbox(label="Identification time")
                transcription_time = gr.Textbox(label="Transcription time")
        with gr.Row(visible=quantized):
            with gr.Column():
                run_quantized_button = gr.Button(value="Run INT8")
            with gr.Column():
                detected_language_quantized = gr.Textbox(label="Detected language ID (Quantized)")
                transcription_quantized = gr.Textbox(label="Transcription (Quantized)")
                identification_time_quantized = gr.Textbox(label="Identification time (Quantized)")
                transcription_time_quantized = gr.Textbox(label="Transcription time (Quantized)")

        run_button.click(
            fn=fn,
            inputs=[audio, gr.Number(0, visible=False)],
            outputs=[
                detected_language,
                transcription,
                identification_time,
                transcription_time,
            ],
        )
        if quantized:
            run_quantized_button.click(
                fn=fn,
                inputs=[audio, gr.Number(1, visible=False)],
                outputs=[
                    detected_language_quantized,
                    transcription_quantized,
                    identification_time_quantized,
                    transcription_time_quantized,
                ],
            )
    return demo
