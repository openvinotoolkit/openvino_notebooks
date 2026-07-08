"""
Gradio helper for MeloTTS with OpenVINO.
Based on the official MeloTTS demo: https://github.com/myshell-ai/MeloTTS
"""

import time
from typing import List

import numpy as np
import gradio as gr

# Example texts shown in the UI, keyed by MeloTTS language code.
EXAMPLE_TEXTS = {
    "ZH": "我最近在学习MeloTTS,使用OpenVINO加速推理,效果非常不错!",
    "EN": "Hello! Welcome to the MeloTTS demo accelerated by OpenVINO Runtime.",
    "ES": "¡Hola! Bienvenido a la demostración de MeloTTS acelerada por OpenVINO.",
    "FR": "Bonjour ! Bienvenue dans la démo MeloTTS accélérée par OpenVINO.",
    "JP": "こんにちは!OpenVINOで高速化されたMeloTTSのデモへようこそ。",
    "KR": "안녕하세요! OpenVINO로 가속된 MeloTTS 데모에 오신 것을 환영합니다.",
}


def make_demo(ov_model, language: str = "ZH"):
    """
    Create a Gradio demo for MeloTTS with OpenVINO.

    Args:
        ov_model: A loaded ``melo_openvino.api.TTS`` instance.
        language: MeloTTS language code of the loaded model (e.g. "ZH", "EN").

    Returns:
        Gradio Blocks demo.
    """

    speakers: List[str] = list(ov_model.hps.data.spk2id.keys())
    sample_rate = ov_model.hps.data.sampling_rate
    default_text = EXAMPLE_TEXTS.get(language.split("_")[0], EXAMPLE_TEXTS["EN"])

    def synthesize(text, speaker, speed, progress=gr.Progress(track_tqdm=True)):
        """Run MeloTTS OpenVINO inference and return audio + status."""
        if not text or not text.strip():
            return None, "Error: Text is required."
        if not speaker:
            return None, "Error: Speaker is required."

        try:
            speaker_id = ov_model.hps.data.spk2id[speaker]

            start_time = time.time()
            audio = ov_model.tts_to_file(
                text.strip(),
                speaker_id,
                output_path=None,
                speed=float(speed),
                quiet=True,
            )
            inference_time = time.time() - start_time

            audio = np.asarray(audio, dtype=np.float32)
            audio_duration = len(audio) / sample_rate

            status = (
                f"✓ Generation completed!\n"
                f"Inference time: {inference_time:.2f}s | "
                f"Audio duration: {audio_duration:.2f}s | "
                f"RTF: {inference_time / max(audio_duration, 0.1):.3f}"
            )

            return (sample_rate, audio), status
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = """
    .gradio-container {max-width: none !important;}
    """

    with gr.Blocks(theme=theme, css=css, title="MeloTTS with OpenVINO") as demo:
        gr.Markdown("""
# MeloTTS with OpenVINO

**Accelerated by OpenVINO™ Runtime**

MeloTTS is a high-quality multilingual text-to-speech library based on VITS.
This demo runs the exported OpenVINO IR for accelerated inference on CPU or GPU.
""")

        with gr.Row():
            with gr.Column(scale=2):
                tts_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=4,
                    placeholder="Enter the text you want to convert to speech...",
                    value=default_text,
                )
                with gr.Row():
                    tts_speaker = gr.Dropdown(
                        label="Speaker",
                        choices=speakers,
                        value=speakers[0] if speakers else None,
                        interactive=True,
                    )
                    tts_speed = gr.Slider(
                        label="Speed",
                        minimum=0.5,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        interactive=True,
                    )
                tts_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Column(scale=2):
                tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                tts_status = gr.Textbox(label="Status", lines=2, interactive=False)

        tts_btn.click(
            synthesize,
            inputs=[tts_text, tts_speaker, tts_speed],
            outputs=[tts_audio_out, tts_status],
        )

        gr.Markdown("""
---
**Links:** [MeloTTS on GitHub](https://github.com/myshell-ai/MeloTTS) | [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
""")

    return demo
