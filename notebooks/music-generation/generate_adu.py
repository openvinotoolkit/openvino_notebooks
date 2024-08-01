# generate_audio.py
import warnings
from functools import partial
import torch
import openvino as ov
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from packaging.version import parse
import importlib.metadata as importlib_metadata
from collections import namedtuple
import numpy as np
from scipy.io.wavfile import write as write_wav
import io

# Ignore tracing warnings
from torch.jit import TracerWarning
warnings.filterwarnings("ignore", category=TracerWarning)

# Load the pipeline
loading_kwargs = {}
if parse(importlib_metadata.version("transformers")) >= parse("4.40.0"):
    loading_kwargs["attn_implementation"] = "eager"

model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", torchscript=True, return_dict=False, **loading_kwargs)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

sample_length = 8  # seconds
n_tokens = sample_length * model.config.audio_encoder.frame_rate + 3
sampling_rate = model.config.audio_encoder.sampling_rate

model.to("cpu")
model.eval()

def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_attention_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, decoder_delay_pattern_mask=None, guidance_scale=None, **kwargs):
    if decoder_delay_pattern_mask is None:
        (decoder_input_ids, decoder_delay_pattern_mask) = self.decoder.build_delay_pattern_mask(
            decoder_input_ids,
            self.generation_config.pad_token_id,
            max_length=self.generation_config.max_length,
        )

    # Apply the delay pattern mask
    decoder_input_ids = self.decoder.apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask)

    if guidance_scale is not None and guidance_scale > 1:
        # For classifier free guidance, replicate the decoder args across the batch dim (split before sampling)
        decoder_input_ids = decoder_input_ids.repeat((2, 1))
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat((2, 1))

    if past_key_values is not None:
        # Cut decoder_input_ids if past is used
        decoder_input_ids = decoder_input_ids[:, -1:]

    return {
        "input_ids": None,  # Encoder_outputs is defined. input_ids not needed
        "encoder_outputs": encoder_outputs,
        "past_key_values": past_key_values,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
        "use_cache": use_cache,
    }

model.prepare_inputs_for_generation = partial(prepare_inputs_for_generation, model)

def generate(prompt: str):
    inputs = processor(
        text=[prompt],
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=n_tokens)
    waveform = audio_values[0].cpu().numpy().squeeze() * 2**15

    # Convert to WAV file format
    wav_io = io.BytesIO()
    write_wav(wav_io, sampling_rate, waveform.astype(np.int16))
    wav_io.seek(0)  # Reset cursor to the beginning of the BytesIO object

    return wav_io
