from collections import namedtuple
from typing import Union, List
from functools import partial

import torch

from whisper.decoding import DecodingTask, Inference, DecodingOptions, DecodingResult


class OpenVINOInference(Inference):
    """
    Wrapper for inference interface
    """

    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        getting logits for given tokens sequence and audio features and save kv_cache

        Parameters:
          tokens: input tokens
          audio_features: input audio features
        Returns:
          logits: predicted by decoder logits
        """
        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]
        logits, self.kv_cache = self.model.decoder(
            tokens, audio_features, kv_cache=self.kv_cache)
        return logits

    def cleanup_caching(self):
        """
        Reset kv_cache to initial state
        """
        self.kv_cache = {}

    def rearrange_kv_cache(self, source_indices):
        """
        Update hidden states cache for selected sequences
        Parameters:
          source_indicies: sequences indicies
        Returns:
          None
        """
        for module, tensor in self.kv_cache.items():
            # update the key/value cache to contain the selected sequences
            self.kv_cache[module] = tensor[source_indices]


class OpenVINODecodingTask(DecodingTask):
    """
    Class for decoding using OpenVINO
    """

    def __init__(self, model: "Whisper", options: DecodingOptions):
        super().__init__(model, options)
        self.inference = OpenVINOInference(model, len(self.initial_tokens))


def patch_whisper_for_ov_inference(model):
    @torch.no_grad()
    def decode(model: "Whisper", mel: torch.Tensor, options: DecodingOptions = DecodingOptions()) -> Union[
        DecodingResult, List[DecodingResult]]:
        """
        Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

        Parameters
        ----------
        model: Whisper
            the Whisper model instance

        mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
            A tensor containing the Mel spectrogram(s)

        options: DecodingOptions
            A dataclass that contains all necessary options for decoding 30-second segments

        Returns
        -------
        result: Union[DecodingResult, List[DecodingResult]]
            The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
        """
        single = mel.ndim == 2
        if single:
            mel = mel.unsqueeze(0)

        result = OpenVINODecodingTask(model, options).run(mel)

        if single:
            result = result[0]

        return result

    Parameter = namedtuple('Parameter', ['device'])

    def parameters():
        return iter([Parameter(torch.device('cpu'))])

    def logits(model, tokens: torch.Tensor, audio_features: torch.Tensor):
        """
        Override for logits extraction method
        Parameters:
          tokens: input tokens
          audio_features: input audio features
        Returns:
          logits: decoder predicted logits
        """
        return model.decoder(tokens, audio_features, None)[0]

    model.parameters = parameters
    model.decode = partial(decode, model)
    model.logits = partial(logits, model)