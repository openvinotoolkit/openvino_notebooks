import torch
from optimum.intel.openvino import OVModelForCausalLM

try:
    from outetts.version.v1.interface import InterfaceHF
    from outetts.version.v1.prompt_processor import PromptProcessor
    from outetts.version.v1.model import HFModel
    from outetts.wav_tokenizer.audio_codec import AudioCodec

    updated_version = True
except ImportError:
    from outetts.v0_1.interface import InterfaceHF
    from outetts.v0_1.audio_codec import AudioCodec
    from outetts.v0_1.prompt_processor import PromptProcessor
    from outetts.v0_1.model import HFModel

    updated_version = False


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
        self.prompt_processor = PromptProcessor(model_path) if not updated_version else PromptProcessor(model_path, ["en"])
        self.model = OVHFModel(model_path, device)
        self.language = "en"
        self.verbose = False
        self.languages = ["en"]
        self._device = torch.device("cpu")
