from outetts.v0_1.interface import InterfaceHF
from outetts.v0_1.audio_codec import AudioCodec
from outetts.v0_1.prompt_processor import PromptProcessor
from outetts.v0_1.model import HFModel
import torch
from optimum.intel.openvino import OVModelForCausalLM


class OVHFModel(HFModel):
    def __init__(self, model_path, device):
        self.device = torch.device("cpu")
        self.model = OVModelForCausalLM.from_pretrained(model_path, device=device)


class InterfaceOV(InterfaceHF):
    def __init__(
        self,
        model_path: str,
        device: str = None,
    ) -> None:
        self.device = torch.device("cpu")
        self.audio_codec = AudioCodec(self.device)
        self.prompt_processor = PromptProcessor(model_path)
        self.model = OVHFModel(model_path, device)
