from pathlib import Path

import torch
import torch.nn as nn
from diffusers import UNet3DConditionModel
from transformers import CLIPTextModel
from diffusers.models.vae import Decoder

from openvino.tools import mo
from openvino.runtime import serialize
from openvino.runtime.utils.data_helpers.wrappers import OVDict


class PyTorchOpenVinoModelConverter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert(self, model: nn.Module, name: str = "model", **convert_kwargs) -> Path:
        xml_path = self.output_dir / f"{name}.xml"
        if not xml_path.exists():
            converted_model = mo.convert_model(model, **convert_kwargs)
            serialize(converted_model, xml_path)
        return xml_path


def convert_unet(unet: UNet3DConditionModel) -> Path:
    converter = PyTorchOpenVinoModelConverter("models/unet")

    return converter.convert(
        unet,
        example_input={
            "sample": torch.randn(2, 4, 16, 32, 32),
            "timestep": torch.tensor(1),
            "encoder_hidden_states": torch.randn(2, 77, 1024),
        },
    )


def convert_vae_modules(decoder: Decoder, post_quant_conv: nn.Conv2d):
    converter = PyTorchOpenVinoModelConverter("models/vae")

    decoder_xml_path = converter.convert(decoder, "decoder", example_input=torch.randn(16, 4, 32, 32))
    post_quant_conv_xml_path = converter.convert(
        post_quant_conv, "post_quant_conv", example_input=torch.randn(16, 4, 32, 32)
    )

    return (decoder_xml_path, post_quant_conv_xml_path)

def convert_text_encoder(encoder: CLIPTextModel):
    converter = PyTorchOpenVinoModelConverter("models/text_encoder")
    xml_path = converter.convert(encoder, example_input=torch.ones(1, 77, dtype=torch.int64))
    
    return xml_path
