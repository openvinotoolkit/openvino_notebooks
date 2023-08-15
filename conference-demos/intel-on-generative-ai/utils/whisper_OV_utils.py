import torch
from openvino.runtime import Core, Tensor
from pathlib import Path
from typing import Optional, Union, List, Dict
from functools import partial
import numpy as np
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

def logits(model, tokens: torch.Tensor, audio_features: torch.Tensor):
    """
    Override for logits extraction method
    Parameters:
      toekns: input tokens
      audio_features: input audio features
    Returns:
      logits: decoder predicted logits
    """
    return model.decoder(tokens, audio_features, None)[0]

def parameters():
    Parameter = namedtuple('Parameter', ['device'])
    return iter([Parameter(torch.device('cpu'))])

@torch.no_grad()
def decode(model: "Whisper", mel: torch.Tensor, options: DecodingOptions = DecodingOptions()) -> Union[DecodingResult, List[DecodingResult]]:
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

class OpenVINOAudioEncoder(torch.nn.Module):
    """
    Helper for inference Whisper encoder model with OpenVINO
    """

    def __init__(self, core, model_path, device='CPU'):
        super().__init__()
        self.model = core.read_model(model_path)
        self.compiled_model = core.compile_model(self.model, device)
        self.output_blob = self.compiled_model.output(0)

    def forward(self, mel: torch.Tensor):
        """
        Inference OpenVINO whisper encoder model.

        Parameters:
          mel: input audio fragment mel spectrogram.
        Returns:
          audio_features: torch tensor with encoded audio features.
        """
        return torch.from_numpy(self.compiled_model(mel)[self.output_blob])

class OpenVINOTextDecoder(torch.nn.Module):
    """
    Helper for inference OpenVINO decoder model
    """

    def __init__(self, core: Core, model_path: Path, device: str = 'CPU'):
        super().__init__()
        self._core = core
        self.model = core.read_model(model_path)
        self._input_names = [inp.any_name for inp in self.model.inputs]
        self.compiled_model = core.compile_model(self.model, device)
        self.device = device

    def init_past_inputs(self, feed_dict):
        """
        Initialize cache input for first step.

        Parameters:
          feed_dict: Dictonary with inputs for inference
        Returns:
          feed_dict: updated feed_dict
        """
        beam_size = feed_dict['tokens'].shape[0]
        audio_len = feed_dict['audio_features'].shape[2]
        previous_seq_len = 0
        for name in self._input_names:
            if name in ['tokens', 'audio_features']:
                continue
            feed_dict[name] = Tensor(np.zeros(
                (beam_size, previous_seq_len, audio_len), dtype=np.float32))
        return feed_dict

    def preprocess_kv_cache_inputs(self, feed_dict, kv_cache):
        """
        Transform kv_cache to inputs

        Parameters:
          feed_dict: dictionary with inputs for inference
          kv_cache: dictionary with cached attention hidden states from previous step
        Returns:
          feed_dict: updated feed dictionary with additional inputs
        """
        if not kv_cache:
            return self.init_past_inputs(feed_dict)
        for k, v in kv_cache.items():
            new_k = f'in_{k}'
            if new_k in self._input_names:
                feed_dict[new_k] = Tensor(v.numpy())
        return feed_dict

    def postprocess_outputs(self, outputs):
        """
        Transform model output to format expected by the pipeline

        Parameters:
          outputs: outputs: raw inference results.
        Returns:
          logits: decoder predicted token logits
          kv_cache: cached attention hidden states
        """
        logits = None
        kv_cache = {}
        for output_t, out in outputs.items():
            if 'logits' in output_t.get_names():
                logits = torch.from_numpy(out)
            else:
                tensor_name = output_t.any_name
                kv_cache[tensor_name.replace(
                    'out_', '')] = torch.from_numpy(out)
        return logits, kv_cache

    def forward(self, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[dict] = None):
        """
        Inference decoder model.

        Parameters:
          x: torch.LongTensor, shape = (batch_size, <= n_ctx) the text tokens
          xa: torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
             the encoded audio features to be attended on
          kv_cache: Dict[str, torch.Tensor], attention modules hidden states cache from previous steps
        Returns:
          logits: decoder predicted logits
          kv_cache: updated kv_cache with current step hidden states
        """
        feed_dict = {'tokens': Tensor(x.numpy()), 'audio_features': Tensor(xa.numpy())}
        feed_dict = (self.preprocess_kv_cache_inputs(feed_dict, kv_cache))
        res = self.compiled_model(feed_dict)
        return self.postprocess_outputs(res)