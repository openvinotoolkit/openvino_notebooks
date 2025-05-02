import gradio as gr
import numpy as np
from encodec import compress, decompress
from encodec.utils import convert_audio
import torch


def make_demo(model, model_sr, model_channels):
    def preprocess(input, sample_rate, model_sr, model_channels):
        input = torch.tensor(input, dtype=torch.float32)
        input = input / 2**15  # adjust to int16 scale
        input = input.unsqueeze(0)
        input = convert_audio(input, sample_rate, model_sr, model_channels)

        return input

    def postprocess(output):
        output = output.squeeze()
        output = output * 2**15  # adjust to [-1, 1] scale
        output = output.numpy(force=True)
        output = output.astype(np.int16)

        return output

    def _compress(input: tuple[int, np.ndarray]):
        sample_rate, waveform = input
        waveform = preprocess(waveform, sample_rate, model_sr, model_channels)

        b = compress(model, waveform, use_lm=False)
        out, out_sr = decompress(b)

        out = postprocess(out)
        return out_sr, out

    demo = gr.Interface(_compress, "audio", "audio", examples=["test_24k.wav"])
    return demo
