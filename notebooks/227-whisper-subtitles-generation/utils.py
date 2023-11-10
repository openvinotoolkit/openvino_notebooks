from collections import namedtuple
from functools import partial
import openvino as ov
from pathlib import Path
from typing import List, Optional, Union

import io
from scipy.io import wavfile
from moviepy.editor import VideoFileClip

import numpy as np
import torch

from whisper.decoding import DecodingTask, Inference, DecodingOptions, DecodingResult


class OpenVINOAudioEncoder(torch.nn.Module):
    """
    Helper for inference Whisper encoder model with OpenVINO
    """

    def __init__(self, core:ov.Core, model_path: Path, device='CPU'):
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

    def __init__(self, core: ov.Core, model_path: Path, device: str = 'CPU'):
        super().__init__()
        self._core = core
        self.model = core.read_model(model_path)
        self._input_names = [inp.any_name for inp in self.model.inputs]
        self.compiled_model = core.compile_model(self.model, device)
        self.device = device
        self.blocks = []

    def init_past_inputs(self, feed_dict):
        """
        Initialize cache input for first step.

        Parameters:
          feed_dict: Dictonary with inputs for inference
        Returns:
          feed_dict: updated feed_dict
        """
        beam_size = feed_dict['x'].shape[0]
        audio_len = feed_dict['xa'].shape[2]
        previous_seq_len = 0
        for name in self._input_names:
            if name in ['x', 'xa']:
                continue
            feed_dict[name] = ov.Tensor(np.zeros((beam_size, previous_seq_len, audio_len), dtype=np.float32))
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
        for k, v in zip(self._input_names[2:], kv_cache):
            feed_dict[k] = ov.Tensor(v)
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
        logits = torch.from_numpy(outputs[0])
        kv_cache = list(outputs.values())[1:]
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
        feed_dict = {'x': ov.Tensor(x.numpy()), 'xa': ov.Tensor(xa.numpy())}
        feed_dict = (self.preprocess_kv_cache_inputs(feed_dict, kv_cache))
        res = self.compiled_model(feed_dict)
        return self.postprocess_outputs(res)


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
            self.kv_cache[module] = tensor[source_indices].detach()


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


def resample(audio, src_sample_rate, dst_sample_rate):
    """
    Resample audio to specific sample rate

    Parameters:
      audio: input audio signal
      src_sample_rate: source audio sample rate
      dst_sample_rate: destination audio sample rate
    Returns:
      resampled_audio: input audio signal resampled with dst_sample_rate
    """
    if src_sample_rate == dst_sample_rate:
        return audio
    duration = audio.shape[0] / src_sample_rate
    resampled_data = np.zeros(shape=(int(duration * dst_sample_rate)), dtype=np.float32)
    x_old = np.linspace(0, duration, audio.shape[0], dtype=np.float32)
    x_new = np.linspace(0, duration, resampled_data.shape[0], dtype=np.float32)
    resampled_audio = np.interp(x_new, x_old, audio)
    return resampled_audio.astype(np.float32)


def audio_to_float(audio):
    """
    convert audio signal to floating point format
    """
    return audio.astype(np.float32) / np.iinfo(audio.dtype).max


def get_audio(video_file):
    """
    Extract audio signal from a given video file, then convert it to float,
    then mono-channel format and resample it to the expected sample rate

    Parameters:
        video_file: path to input video file
    Returns:
      resampled_audio: mono-channel float audio signal with 16000 Hz sample rate
                       extracted from video
    """
    input_video = VideoFileClip(str(video_file))
    input_video.audio.write_audiofile(video_file.stem + '.wav', verbose=False, logger=None)
    input_audio_file = video_file.stem + '.wav'
    sample_rate, audio = wavfile.read(
        io.BytesIO(open(input_audio_file, 'rb').read()))
    audio = audio_to_float(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # The model expects mono-channel audio with a 16000 Hz sample rate, represented in floating point range. When the
    # audio from the input video does not meet these requirements, we will need to apply preprocessing.
    resampled_audio = resample(audio, sample_rate, 16000)
    return resampled_audio


def format_timestamp(seconds: float):
    """
    format time in srt-file expected format
    """
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return (f"{hours}:" if hours > 0 else "00:") + f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def prepare_srt(transcription):
    """
    Format transcription into srt file format
    """
    segment_lines = []
    for segment in transcription["segments"]:
        segment_lines.append(str(segment["id"] + 1) + "\n")
        time_start = format_timestamp(segment["start"])
        time_end = format_timestamp(segment["end"])
        time_str = f"{time_start} --> {time_end}\n"
        segment_lines.append(time_str)
        segment_lines.append(segment["text"] + "\n\n")
    return segment_lines
