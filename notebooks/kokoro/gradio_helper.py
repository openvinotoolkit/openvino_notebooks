from pathlib import Path

import gradio as gr
import numpy as np
import openvino as ov
import requests


def make_demo(pipeline, model_dir: Path):
    model_dir = Path(model_dir)
    speaker_embedding_shape = pipeline.get_speaker_embedding_shape()

    def load_voice(voice: str) -> ov.Tensor:
        voice_path = model_dir / "voices" / f"{voice}.bin"
        data = np.fromfile(voice_path, dtype=np.float32)
        expected_size = int(np.prod(speaker_embedding_shape))
        if data.size != expected_size:
            raise ValueError(f"Voice embedding has {data.size} values, but the model expects " f"{expected_size} for shape {tuple(speaker_embedding_shape)}")
        return ov.Tensor(data.reshape(speaker_embedding_shape))

    def get_audio_data(speech: ov.Tensor) -> np.ndarray:
        try:
            return np.asarray(speech.data, dtype=np.float32).reshape(-1)
        except RuntimeError:
            host_tensor = ov.Tensor(speech.element_type, speech.shape)
            speech.copy_to(host_tensor)
            return np.asarray(host_tensor.data, dtype=np.float32).reshape(-1)

    def generate(text: str, voice: str = "af_heart", speed: float = 1.0):
        speaker_embedding = load_voice(voice)
        result = pipeline.generate(
            text,
            speaker_embedding,
            language="en-us",
            speed=float(speed),
        )
        audio = get_audio_data(result.speeches[0])
        return result.output_sample_rate, audio

    if not Path("en.txt").exists():
        r = requests.get("https://huggingface.co/spaces/hexgrad/Kokoro-TTS/raw/main/en.txt", timeout=30)
        r.raise_for_status()
        with open("en.txt", "w") as f:
            f.write(r.text)
    with open("en.txt", "r") as r:
        random_quotes = [line.strip() for line in r]

    def get_random_quote():
        return str(np.random.choice(random_quotes))

    CHOICES = {
        "🇺🇸 🚺 Heart ❤️": "af_heart",
        "🇺🇸 🚺 Bella 🔥": "af_bella",
        "🇺🇸 🚺 Nicole 🎧": "af_nicole",
        "🇺🇸 🚺 Aoede": "af_aoede",
        "🇺🇸 🚺 Kore": "af_kore",
        "🇺🇸 🚺 Sarah": "af_sarah",
        "🇺🇸 🚺 Nova": "af_nova",
        "🇺🇸 🚺 Sky": "af_sky",
        "🇺🇸 🚺 Alloy": "af_alloy",
        "🇺🇸 🚺 Jessica": "af_jessica",
        "🇺🇸 🚺 River": "af_river",
        "🇺🇸 🚹 Michael": "am_michael",
        "🇺🇸 🚹 Fenrir": "am_fenrir",
        "🇺🇸 🚹 Puck": "am_puck",
        "🇺🇸 🚹 Echo": "am_echo",
        "🇺🇸 🚹 Eric": "am_eric",
        "🇺🇸 🚹 Liam": "am_liam",
        "🇺🇸 🚹 Onyx": "am_onyx",
        "🇺🇸 🚹 Santa": "am_santa",
        "🇺🇸 🚹 Adam": "am_adam",
    }

    BANNER_TEXT = """
    [***Kokoro*** **is an open-weight TTS model with 82 million parameters.**](https://huggingface.co/hexgrad/Kokoro-82M)
    This OpenVINO GenAI demo showcases American English voices.
    """
    with gr.Blocks() as app:
        gr.Markdown(BANNER_TEXT, container=True)
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Input Text",
                    info="Long text is split into supported phoneme-length chunks automatically.",
                )
                voice = gr.Dropdown(
                    list(CHOICES.items()),
                    value="af_heart",
                    label="Voice",
                )
                speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label="Speed")
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    random_btn = gr.Button("🎲 Random Quote 💬", variant="secondary")
                out_audio = gr.Audio(label="Output Audio", interactive=False, autoplay=True)
        random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
        generate_btn.click(fn=generate, inputs=[text, voice, speed], outputs=[out_audio])

    return app
