from typing import Callable
import gradio as gr
import openvino as ov


examples = [
    "acoustic folk violin jam",
    "bossa nova with distorted guitar",
    "arabic gospel vocals",
    "piano funk",
    "swing jazz trumpet",
    "jamaican dancehall vocals",
    "ibiza at 3am",
    "k-pop boy group",
    "laughing",
    "water drops",
]


def make_demo(generate_fn: Callable, select_device_fn: Callable):
    available_devices = ov.Core().available_devices + ["AUTO"]
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Markdown("# Riffusion music generation with OpenVINO\n" " Describe a musical prompt, generate music by getting a spectrogram image and sound.")

            prompt_input = gr.Textbox(placeholder="", label="Musical prompt")
            negative_prompt = gr.Textbox(label="Negative prompt")
            device = gr.Dropdown(choices=available_devices, value="AUTO", label="Device")

            send_btn = gr.Button(value="Get a new spectrogram!")
            gr.Examples(examples, prompt_input, examples_per_page=15)

        with gr.Column():
            sound_output = gr.Audio(type="filepath", label="spectrogram sound")
            spectrogram_output = gr.Image(label="spectrogram image result", height=256)

        send_btn.click(
            fn=generate_fn,
            inputs=[prompt_input, negative_prompt],
            outputs=[spectrogram_output, sound_output],
        )
        device.change(select_device_fn, [device, prompt_input], [prompt_input])
    return demo
