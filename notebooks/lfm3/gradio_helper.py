from typing import Callable, Optional

import gradio as gr
import numpy as np


def make_demo(generate_fn: Callable, example_image: Optional[str] = None):
    """Gradio demo for LFM3-Omni.

    generate_fn(text, image, audio, speak) must return (reply_text, speech) where `audio` is a
    (sampling_rate, np.ndarray) pair or None and `speech` is a (24000, np.ndarray) pair or None.
    """

    def respond(text, image, audio, output_mode):
        text = (text or "").strip()
        if not text and image is None and audio is None:
            raise gr.Error("Provide a text prompt, an image or a recorded question.")
        reply, speech = generate_fn(text, image, audio, output_mode == "Text and speech")
        return reply, speech

    with gr.Blocks(title="LFM3-Omni with OpenVINO") as demo:
        gr.Markdown("# LFM3-Omni with OpenVINO\nAsk about an image by text or by voice and get the answer as text or as text and speech.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image (optional)")
                audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Spoken question (optional)")
                text = gr.Textbox(label="Text prompt (optional)", lines=2)
                output_mode = gr.Radio(["Text", "Text and speech"], value="Text", label="Answer as")
                submit = gr.Button("Submit", variant="primary")
            with gr.Column():
                reply = gr.Textbox(label="Answer", lines=10)
                speech = gr.Audio(label="Spoken answer", type="numpy", autoplay=True)
        if example_image is not None:
            gr.Examples([[example_image, "What is unusual in this picture?", "Text"]], inputs=[image, text, output_mode])
        submit.click(respond, inputs=[text, image, audio, output_mode], outputs=[reply, speech])
    return demo


def to_mono_float32(audio):
    """Converts a Gradio (sampling_rate, int16/float array) pair into a mono float32 waveform in [-1, 1]."""
    sampling_rate, waveform = audio
    waveform = np.asarray(waveform)
    if np.issubdtype(waveform.dtype, np.integer):
        waveform = waveform.astype(np.float32) / np.iinfo(waveform.dtype).max
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return sampling_rate, waveform.astype(np.float32)
