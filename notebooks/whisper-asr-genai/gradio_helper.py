from pathlib import Path
from transformers.pipelines.audio_utils import ffmpeg_read

from typing import Callable
import gradio as gr
import requests
import time

audio_en_example_path = Path("en_example.wav")
audio_ml_example_path = Path("ml_example.wav")

if not audio_en_example_path.exists():
    r = requests.get("https://huggingface.co/spaces/distil-whisper/whisper-vs-distil-whisper/resolve/main/assets/example_1.wav")
    with open(audio_en_example_path, "wb") as f:
        f.write(r.content)


if not audio_ml_example_path.exists():
    r = requests.get("https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jeanNL.wav")
    with open(audio_ml_example_path, "wb") as f:
        f.write(r.content)


MAX_AUDIO_MINS = 30  # maximum audio input in minutes


class GradioPipeline:
    def __init__(self, ov_pipe, model_id, quantized=False) -> None:
        self.pipe = ov_pipe
        self.model_id = model_id
        self.multilingual = not model_id.endswith(".en")
        self.quantized = quantized

    def forward(self, inputs, task="transcribe", language="auto"):
        generate_kwargs = {}
        if not self.multilingual and task != "Transcribe":
            raise gr.Error("The model only supports English. The task 'translate' could not be applied.")
        elif task == "Translate":
            generate_kwargs = {"task": "translate"}
            if language and language != "auto":
                generate_kwargs["language"] = language

        if inputs is None:
            raise gr.Error("No audio file submitted! Please record or upload an audio file before submitting your request.")

        with open(inputs, "rb") as f:
            inputs = f.read()

        inputs = ffmpeg_read(inputs, 16000)
        audio_length_mins = len(inputs) / 16000 / 60

        if audio_length_mins > MAX_AUDIO_MINS:
            raise gr.Error(
                f"To ensure fair usage of the Space, the maximum audio length permitted is {MAX_AUDIO_MINS} minutes."
                f"Got an audio of length {round(audio_length_mins, 3)} minutes."
            )

        start_time = time.time()
        ov_text = self.pipe.generate(inputs.copy(), **generate_kwargs)
        ov_time = time.time() - start_time
        ov_time = round(ov_time, 2)

        return ov_text, ov_time


def make_demo(gr_pipeline):
    examples = [[str(audio_en_example_path), ""]]
    if gr_pipeline.multilingual:
        examples.append([str(audio_ml_example_path), "<|fr|>"])

    with gr.Blocks() as demo:
        gr.HTML(
            f"""
                    <div style="text-align: center; max-width: 700px; margin: 0 auto;">
                    <div
                        style="
                        display: grid; align-items: center; gap: 0.8rem; font-size: 1.75rem;
                        "
                    >
                        <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                            OpenVINO Generate API Whisper demo {'with quantized model.' if gr_pipeline.quantized else ''}
                        </h1>
                        <div style="font-size: 12px; {'' if gr_pipeline.multilingual else 'display: none;'}">For task 'Translate', please, find the avalible languages
                            <a href='https://huggingface.co/{gr_pipeline.model_id}/blob/main/generation_config.json'>in 'generation_config.json' of the model</a>
                            or get 'generation_config' by ov_pipe.get_generation_config() and check the attribute 'lang_to_id'</div>
                    </div>
                    </div>
                """
        )
        audio = gr.components.Audio(type="filepath", label="Audio input")
        language = gr.components.Textbox(
            label="Language.",
            info="List of avalible languages you can find in generation_config.lang_to_id dictionary. Example: <|en|>. Empty string will mean autodetection",
            value="",
            visible=gr_pipeline.multilingual,
        )
        with gr.Row():
            button_transcribe = gr.Button("Transcribe")
            button_translate = gr.Button("Translate", visible=gr_pipeline.multilingual)
        with gr.Row():
            infer_time = gr.components.Textbox(label="OpenVINO Whisper Generation Time (s)")
        with gr.Row():
            result = gr.components.Textbox(label="OpenVINO Whisper Result", show_copy_button=True)
        button_transcribe.click(
            fn=gr_pipeline.forward,
            inputs=[audio, button_transcribe, language],
            outputs=[result, infer_time],
        )
        button_translate.click(
            fn=gr_pipeline.forward,
            inputs=[audio, button_translate, language],
            outputs=[result, infer_time],
        )
        gr.Markdown("## Examples")
        gr.Examples(
            examples,
            inputs=[audio, language],
            outputs=[result, infer_time],
            fn=gr_pipeline.forward,
            cache_examples=False,
        )

    return demo
