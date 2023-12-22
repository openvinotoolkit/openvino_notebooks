import intel_extension_for_pytorch as ipex
import nltk
import numpy as np
from bark import generate_audio, preload_models, SAMPLE_RATE
from bark.generation import models


class OVBark:

    def __init__(self, small_models: bool = False, speaker: str = None):
        preload_models(
            text_use_small=small_models,
            coarse_use_gpu=False,
            coarse_use_small=small_models,
            fine_use_gpu=False,
            fine_use_small=small_models,
            codec_use_gpu=False,
            force_reload=False,
        )
        models["text"]["model"] = ipex.optimize(models["text"]["model"].to("cpu"))
        models["coarse"] = ipex.optimize(models["coarse"].to("cpu"))
        models["fine"] = ipex.optimize(models["fine"].to("cpu"))

        self.history_prompt = None
        if speaker is not None:
            self.history_prompt = "v2/en_speaker_6" if speaker == "male" else "v2/en_speaker_9"

        nltk.download("punkt", quiet=True)

    def generate_audio(
            self,
            text: str,
            text_temp: float = 0.7,
            waveform_temp: float = 0.7,
            silent: bool = False,
    ):
        """Generate audio array from input text.

        Args:
            text: text to be turned into audio
            text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            silent: disable progress bar

        Returns:
            numpy audio array at sample frequency 24khz
        """
        text = text.replace("\n", " ").strip()
        sentences = nltk.sent_tokenize(text)

        pieces = []
        for sentence in sentences:
            audio_array = generate_audio(sentence, self.history_prompt, text_temp, waveform_temp, silent)
            pieces.append(audio_array)
        return np.concatenate(pieces)
