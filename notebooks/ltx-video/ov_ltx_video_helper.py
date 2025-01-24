import gc
from collections import namedtuple
from pathlib import Path

import torch

import openvino as ov
import nncf


def convert(model: torch.nn.Module, xml_path: str, example_input, model_name: str, to_compress_weights):
    xml_path = Path(xml_path)
    if not xml_path.exists():
        print(f"⌛ {model_name} conversion started")
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            converted_model = ov.convert_model(model, example_input=example_input)
        if to_compress_weights:
            converted_model = nncf.compress_weights(converted_model, mode=nncf.CompressWeightsMode.INT8_ASYM)
        ov.save_model(converted_model, xml_path)
        del model
        gc.collect()
        # cleanup memory
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()
        print(f"✅ {model_name} model conversion finished")
    else:
        print(f"✅ Found converted {model_name} model")


def convert_text_encoder(text_encoder, output_dir, to_compress_weights=True):
    example_input = {
        "input_ids": torch.zeros(1, 128, dtype=torch.int64),
    }

    convert(text_encoder, output_dir, example_input, "text_encoder", to_compress_weights)


def convert_transformer(transformer, output_dir, to_compress_weights=True):

    example_input = {
        "hidden_states": torch.rand([2, 2310, 128], dtype=torch.float32),
        "encoder_hidden_states": torch.rand([2, 128, 4096], dtype=torch.float32),
        "timestep": torch.tensor([1000.0, 1000.0]),
        "encoder_attention_mask": torch.ones([2, 128]),
        "num_frames": torch.tensor(7),
        "height": torch.tensor(15),
        "width": torch.tensor(22),
        "rope_interpolation_scale": torch.tensor([0.32, 32, 32]),
    }
    convert(transformer, output_dir, example_input, "transformer", to_compress_weights)


def convert_vae_decoder(vae_decoder, output_dir, to_compress_weights=True):
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents, timestamp):
            return self.vae.decode(latents)

    convert(VAEDecoderWrapper(vae_decoder), output_dir, (torch.rand([1, 128, 7, 15, 22], dtype=torch.float32), torch.tensor(0)), "vae", to_compress_weights)


class ConvTransformerWrapper(torch.nn.Module):
    def __init__(self, transformer, config):
        super().__init__()
        self.transformer = transformer
        self.config = config

    def forward(self, hidden_states, encoder_hidden_states, timestep, encoder_attention_mask, num_frames, height, width, rope_interpolation_scale, **kwargs):
        outputs = self.transformer(
            {
                "hidden_states": hidden_states,
                "encoder_hidden_states": encoder_hidden_states,
                "timestep": timestep,
                "encoder_attention_mask": encoder_attention_mask,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "rope_interpolation_scale": torch.tensor(rope_interpolation_scale),
            },
        )

        return [torch.from_numpy(outputs[0])]


class ConvTransformerWrapper(torch.nn.Module):
    def __init__(self, transformer, config):
        super().__init__()
        self.transformer = transformer
        self.config = config

    def forward(self, hidden_states, encoder_hidden_states, timestep, encoder_attention_mask, num_frames, height, width, rope_interpolation_scale, **kwargs):
        outputs = self.transformer(
            {
                "hidden_states": hidden_states,
                "encoder_hidden_states": encoder_hidden_states,
                "timestep": timestep,
                "encoder_attention_mask": encoder_attention_mask,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "rope_interpolation_scale": torch.tensor(rope_interpolation_scale),
            },
        )

        return [torch.from_numpy(outputs[0])]


class VAEWrapper(torch.nn.Module):
    def __init__(self, vae, config, latents_mean, latents_std):
        super().__init__()
        self.vae = vae
        self.config = config
        self.latents_mean = latents_mean
        self.latents_std = latents_std

    def decode(self, latents=None, timestamp=None, **kwargs):
        inputs = {
            "latents": latents,
            # "timestamp": timestamp
        }

        outs = self.vae(inputs)
        outs = namedtuple("VAE", "sample")(torch.from_numpy(outs[0]))

        return outs


EncoderOutput = namedtuple("EncoderOutput", "last_hidden_state")


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, text_encoder, dtype):
        super().__init__()
        self.text_encoder = text_encoder
        self.dtype = dtype

    def forward(self, input_ids=None, attention_mask=None):
        inputs = {
            "input_ids": input_ids,
        }
        last_hidden_state = self.text_encoder(inputs)[0]
        return EncoderOutput(torch.from_numpy(last_hidden_state))
