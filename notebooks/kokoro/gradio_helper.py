import gradio as gr
import requests
from pathlib import Path
import torch
import numpy as np


def make_demo(pipeline):
    def generate_first(text, voice="af_heart", speed=1):
        pipeline.load_voice(voice)
        for gs, ps, audio in pipeline(text, voice, speed):
            return (24000, audio.numpy()), ps
        return None, ""

    # Arena API
    def predict(text, voice="af_heart", speed=1):
        return generate_first(text, voice, speed, use_gpu=False)[0]

    def tokenize_first(text, voice="af_heart"):
        for _, ps, _ in pipeline(text, voice):
            return ps
        return ""

    def generate_all(text, voice="af_heart", speed=1):
        pipeline.load_voice(voice)
        first = True
        for _, ps, audio in pipeline(text, voice, speed):
            yield 24000, audio.numpy()
            if first:
                first = False
                yield 24000, torch.zeros(1).numpy()

    if not Path("en.txt").exists():
        r = requests.get("https://huggingface.co/spaces/hexgrad/Kokoro-TTS/raw/main/en.txt")
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

    TOKEN_NOTE = """
    💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`
    💬 To adjust intonation, try punctuation `;:,.!?—…"()“”` or stress `ˈ` and `ˌ`
    ⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`
    ⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)
    """

    with gr.Blocks() as generate_tab:
        out_audio = gr.Audio(label="Output Audio", interactive=False, streaming=False, autoplay=True)
        generate_btn = gr.Button("Generate", variant="primary")
        with gr.Accordion("Output Tokens", open=True):
            out_ps = gr.Textbox(interactive=False, show_label=False, info="Tokens used to generate the audio, up to 510 context length.")
            tokenize_btn = gr.Button("Tokenize", variant="secondary")
            gr.Markdown(TOKEN_NOTE)
            predict_btn = gr.Button("Predict", variant="secondary", visible=False)

    STREAM_NOTE = ["⚠️ There is an unknown Gradio bug that might yield no audio the first time you click `Stream`."]
    STREAM_NOTE = "\n\n".join(STREAM_NOTE)

    with gr.Blocks() as stream_tab:
        out_stream = gr.Audio(label="Output Audio Stream", interactive=False, streaming=True, autoplay=True)
        with gr.Row():
            stream_btn = gr.Button("Stream", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")
        with gr.Accordion("Note", open=True):
            gr.Markdown(STREAM_NOTE)

    BANNER_TEXT = """
    [***Kokoro*** **is an open-weight TTS model with 82 million parameters.**](https://huggingface.co/hexgrad/Kokoro-82M)
    This demo only showcases English, but you can directly use the model to access other languages.
    """
    with gr.Blocks() as app:
        with gr.Row():
            gr.Markdown(BANNER_TEXT, container=True)
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Input Text", info=f"Up to ~500 characters per Generate, or '∞' characters per Stream")
                with gr.Row():
                    voice = gr.Dropdown(list(CHOICES.items()), value="af_heart", label="Voice", info="Quality and availability vary by language")
                speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label="Speed")
                random_btn = gr.Button("🎲 Random Quote 💬", variant="secondary")
            with gr.Column():
                gr.TabbedInterface([generate_tab, stream_tab], ["Generate", "Stream"])
        random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
        generate_btn.click(fn=generate_first, inputs=[text, voice, speed], outputs=[out_audio, out_ps])
        tokenize_btn.click(fn=tokenize_first, inputs=[text, voice], outputs=[out_ps])
        stream_event = stream_btn.click(fn=generate_all, inputs=[text, voice, speed], outputs=[out_stream])
        stop_btn.click(fn=None, cancels=stream_event)
        predict_btn.click(fn=predict, inputs=[text, voice, speed], outputs=[out_audio])

    return app
