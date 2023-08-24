import whisper
import logging
from functools import wraps

from datasets import load_dataset
import torch
import pickle
from typing import Optional, Union, List, Dict, Tuple
from functools import partial
import multiprocessing

from openvino.runtime import Core, Tensor
import openvino.tools.mo
import openvino.tools.ovc
import re
from tqdm import tqdm
from jiwer import compute_measures

from transformers import WhisperProcessor
from evaluate import load

import subprocess

import datetime
from pathlib import Path
import numpy as np
import openvino.runtime as ov
import nncf
from nncf import nncf_logger
from nncf.quantization.range_estimator import RangeEstimatorParameters, StatisticsCollectorParameters, StatisticsType, \
    AggregatorType
from nncf.torch import register_module
from nncf import IgnoredScope
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters, OverflowFix

from internal_utils import download_video, get_audio, convert_input_data_to_ov_tensor, convert_input_data_to_np, \
    prepare_srt
from utils import patch_whisper_for_ov_inference
from ignored_scopes import ignored_scope1, ignored_scope2, ignored_scope3

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("whisper")

TASK = "transcribe"
# TASK = "translate"

# OV_COMPILE_MODEL_CONFIG = {"INFERENCE_NUM_THREADS": multiprocessing.cpu_count()}
OV_COMPILE_MODEL_CONFIG = {"INFERENCE_NUM_THREADS": None}


VERBOSE = bool(0)

BASE_DIR = Path("./")

COLLECT_INIT_DATA = bool(0)
encoder_init_data = []
decoder_init_data = []

num_encoder_forwards = 0
num_decoder_forwards = 0

core = Core()


SAVE_CALIBRATION_DATA = bool(0)
LOAD_CALIBRATION_DATA = bool(0)
# CALIBRATION_DATA_CACHE = 'calibration/librispeech_asr_dummy_{}.pkl'
# CALIBRATION_DATASET = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# CALIBRATION_DATASET = reversed(load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"))
CALIBRATION_DATA_CACHE = 'calibration/librispeech_asr_clean_train100_{}.pkl'
CALIBRATION_DATASET = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True).take(30)
# CALIBRATION_DATA_CACHE = 'calibration/common_voice_11_0_{}.pkl'
# CALIBRATION_DATASET = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True).take(100)

start_time = None

wer = load("wer")

OVC_API_ENCODER = bool(1)
OVC_API_DECODER = bool(1)

# ORIGINAL_ENCODER_MODEL_DIR = BASE_DIR / ('original_model_ovc' if OVC_API_ENCODER else 'original_model')
# ORIGINAL_DECODER_MODEL_DIR = BASE_DIR / ('original_model_ovc' if OVC_API_DECODER else 'original_model')
ORIGINAL_ENCODER_MODEL_DIR = BASE_DIR / ('' if OVC_API_ENCODER else 'original_model')
ORIGINAL_DECODER_MODEL_DIR = BASE_DIR / ('' if OVC_API_DECODER else 'original_model')


class OpenVINOAudioEncoder(torch.nn.Module):
    """
    Helper for inference Whisper encoder model with OpenVINO
    """

    def __init__(self, core, model_path, device='CPU'):
        super().__init__()
        self.model = core.read_model(model_path)
        self.compiled_model = core.compile_model(self.model, device, config=OV_COMPILE_MODEL_CONFIG)
        self.output_blob = self.compiled_model.output(0)

    def forward(self, mel: torch.Tensor):
        """
        Inference OpenVINO whisper encoder model.

        Parameters:
          mel: input audio fragment mel spectrogram.
        Returns:
          audio_features: torch tensor with encoded audio features.
        """
        global num_encoder_forwards
        num_encoder_forwards += 1
        if COLLECT_INIT_DATA:
            encoder_init_data.append(mel)
        return torch.from_numpy(self.compiled_model(mel)[self.output_blob])


class OpenVINOTextDecoderOld(torch.nn.Module):
    """
    Helper for inference OpenVINO decoder model
    """

    def __init__(self, core: Core, model: Union[Path, ov.CompiledModel], device: str = 'CPU'):
        super().__init__()
        self._core = core
        if isinstance(model, ov.CompiledModel):
            self.model = self.compiled_model = model
        elif isinstance(model, Path):
            self.model = core.read_model(model)
            self.compiled_model = core.compile_model(self.model, device, config=OV_COMPILE_MODEL_CONFIG)
        else:
            raise Exception
        self._input_names = [inp.any_name for inp in self.model.inputs]
        self.compiled_model = core.compile_model(self.model, device, config=OV_COMPILE_MODEL_CONFIG)
        self.device = device

    def init_past_inputs(self, feed_dict):
        """
        Initialize cache input for first step.

        Parameters:
          feed_dict: Dictionary with inputs for inference
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
        global num_decoder_forwards
        num_decoder_forwards += 1
        feed_dict = {'tokens': Tensor(x.numpy()), 'audio_features': Tensor(xa.numpy())}
        feed_dict = (self.preprocess_kv_cache_inputs(feed_dict, kv_cache))
        if COLLECT_INIT_DATA:
            decoder_init_data.apend(feed_dict)
        if VERBOSE and start_time is not None:
            print(num_encoder_forwards, num_decoder_forwards, feed_dict["tokens"].shape, feed_dict["in_k_0a"].shape,
                  datetime.datetime.now() - start_time)
        res = self.compiled_model(feed_dict)
        return self.postprocess_outputs(res)


class OpenVINOTextDecoderNew(torch.nn.Module):
    """
    Helper for inference OpenVINO decoder model
    """

    def __init__(self, core: Core, model: Union[Path, ov.CompiledModel], device: str = 'CPU'):
        super().__init__()
        self._core = core
        if isinstance(model, ov.CompiledModel):
            self.model = self.compiled_model = model
        elif isinstance(model, Path):
            self.model = core.read_model(model)
            self.compiled_model = core.compile_model(self.model, device, config=OV_COMPILE_MODEL_CONFIG)
        else:
            raise Exception
        self._input_names = [inp.any_name for inp in self.model.inputs]
        self.device = device

    def init_past_inputs(self, feed_dict):
        """
        Initialize cache input for first step.

        Parameters:
          feed_dict: Dictionary with inputs for inference
        Returns:
          feed_dict: updated feed_dict
        """
        beam_size = feed_dict['x'].shape[0]
        audio_len = feed_dict['xa'].shape[2]
        previous_seq_len = 0
        for name in self._input_names:
            if name in ['x', 'xa']:
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
        for k, v in zip(self._input_names[2:], kv_cache):
            feed_dict[k] = Tensor(v)
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
        global num_decoder_forwards
        num_decoder_forwards += 1
        feed_dict = {'x': Tensor(x.numpy()), 'xa': Tensor(xa.numpy())}
        feed_dict = (self.preprocess_kv_cache_inputs(feed_dict, kv_cache))
        if COLLECT_INIT_DATA:
            decoder_init_data.append(feed_dict)
        if VERBOSE and start_time is not None:
            print(num_encoder_forwards, num_decoder_forwards, feed_dict["tokens"].shape, feed_dict["in_k_0a"].shape,
                  datetime.datetime.now() - start_time)
        res = self.compiled_model(feed_dict)
        return self.postprocess_outputs(res)


OpenVINOTextDecoder = OpenVINOTextDecoderNew if OVC_API_DECODER else OpenVINOTextDecoderOld


def patch_whisper_decoder_for_export(decoder):
    positional_embeddings_size = decoder.positional_embedding.shape[0]

    def save_to_cache(cache: Dict[str, torch.Tensor], module: str, output: torch.Tensor):
        """
        Saving cached attention hidden states for previous tokens.
        Parameters:
          cache: dictionary with cache.
          module: current attention module name.
          output: predicted hidden state.
        Returns:
          output: cached attention hidden state for specified attention module.
        """
        if module not in cache or output.shape[1] > positional_embeddings_size:
            # save as-is, for the first token or cross attention
            cache[module] = output
        else:
            cache[module] = torch.cat([cache[module], output], dim=1).detach()
        return cache[module]

    def attention_forward(
            attention_module,
            x: torch.Tensor,
            xa: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[dict] = None,
            idx: int = 0
    ):
        """
        Override for forward method of decoder attention module with storing cache values explicitly.
        Parameters:
          attention_module: current attention module
          x: input token ids.
          xa: input audio features (Optional).
          mask: mask for applying attention (Optional).
          kv_cache: dictionary with cached key values for attention modules.
          idx: idx for search in kv_cache.
        Returns:
          attention module output tensor
          updated kv_cache
        """
        q = attention_module.query(x)

        if kv_cache is None or xa is None:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = attention_module.key(x if xa is None else xa)
            v = attention_module.value(x if xa is None else xa)
            if kv_cache is not None:
                k = save_to_cache(kv_cache, f'k_{idx}', k)
                v = save_to_cache(kv_cache, f'v_{idx}', v)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache.get(f'k_{idx}', save_to_cache(
                kv_cache, f'k_{idx}', attention_module.key(xa)))
            v = kv_cache.get(f'v_{idx}', save_to_cache(
                kv_cache, f'v_{idx}', attention_module.value(xa)))

        wv, qk = attention_module.qkv_attention(q, k, v, mask)
        return attention_module.out(wv), kv_cache

    def block_forward(
            residual_block,
            x: torch.Tensor,
            xa: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[dict] = None,
            idx: int = 0
    ):
        """
        Override for residual block forward method for providing kv_cache to attention module.
          Parameters:
            residual_block: current residual block.
            x: input token_ids.
            xa: input audio features (Optional).
            mask: attention mask (Optional).
            kv_cache: cache for storing attention key values.
            idx: index of current residual block for search in kv_cache.
          Returns:
            x: residual block output
            kv_cache: updated kv_cache

        """
        x0, kv_cache = residual_block.attn(residual_block.attn_ln(
            x), mask=mask, kv_cache=kv_cache, idx=f'{idx}a')
        x = x + x0
        if residual_block.cross_attn:
            x1, kv_cache = residual_block.cross_attn(
                residual_block.cross_attn_ln(x), xa, kv_cache=kv_cache, idx=f'{idx}c')
            x = x + x1
        x = x + residual_block.mlp(residual_block.mlp_ln(x))
        return x, kv_cache

    # update forward functions
    for idx, block in enumerate(decoder.blocks):
        block.forward = partial(block_forward, block, idx=idx)
        block.attn.forward = partial(attention_forward, block.attn)
        if block.cross_attn:
            block.cross_attn.forward = partial(attention_forward, block.cross_attn)

    def decoder_forward(decoder_, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[dict] = None):
        """
        Override for decoder forward method.
        Parameters:
          x: torch.LongTensor, shape = (batch_size, <= n_ctx) the text tokens
          xa: torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
               the encoded audio features to be attended on
          kv_cache: Dict[str, torch.Tensor], attention modules hidden states cache from previous steps
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = decoder_.token_embedding(
            x) + decoder.positional_embedding[offset: offset + x.shape[-1]]
        x = x.to(xa.dtype)

        for block in decoder_.blocks:
            x, kv_cache = block(x, xa, mask=decoder_.mask, kv_cache=kv_cache)

        x = decoder_.ln(x)
        # logits = (x @ torch.transpose(decoder_.token_embedding.weight.to(x.dtype), 1, 0)).float()
        logits = decoder_.linear(x).float()

        return logits, kv_cache

    # override decoder forward
    decoder.forward = partial(decoder_forward, decoder)


def patch_whisper_decoder_for_export_ovc(decoder):
    def attention_forward(
            attention_module,
            x: torch.Tensor,
            xa: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Override for forward method of decoder attention module with storing cache values explicitly.
        Parameters:
          attention_module: current attention module
          x: input token ids.
          xa: input audio features (Optional).
          mask: mask for applying attention (Optional).
          kv_cache: dictionary with cached key values for attention modules.
          idx: idx for search in kv_cache.
        Returns:
          attention module output tensor
          updated kv_cache
        """
        q = attention_module.query(x)

        if xa is None:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = attention_module.key(x)
            v = attention_module.value(x)
            if kv_cache is not None:
                k = torch.cat((kv_cache[0], k), dim=1)
                v = torch.cat((kv_cache[1], v), dim=1)
            kv_cache_new = (k, v)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = attention_module.key(xa)
            v = attention_module.value(xa)
            kv_cache_new = (None, None)

        wv, qk = attention_module.qkv_attention(q, k, v, mask)
        return attention_module.out(wv), kv_cache_new

    def block_forward(
            residual_block,
            x: torch.Tensor,
            xa: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Override for residual block forward method for providing kv_cache to attention module.
          Parameters:
            residual_block: current residual block.
            x: input token_ids.
            xa: input audio features (Optional).
            mask: attention mask (Optional).
            kv_cache: cache for storing attention key values.
          Returns:
            x: residual block output
            kv_cache: updated kv_cache

        """
        x0, kv_cache = residual_block.attn(residual_block.attn_ln(
            x), mask=mask, kv_cache=kv_cache)
        x = x + x0
        if residual_block.cross_attn:
            x1, _ = residual_block.cross_attn(
                residual_block.cross_attn_ln(x), xa)
            x = x + x1
        x = x + residual_block.mlp(residual_block.mlp_ln(x))
        return x, kv_cache

    # update forward functions
    for idx, block in enumerate(decoder.blocks):
        block.forward = partial(block_forward, block)
        block.attn.forward = partial(attention_forward, block.attn)
        if block.cross_attn:
            block.cross_attn.forward = partial(attention_forward, block.cross_attn)

    def decoder_forward(decoder_, x: torch.Tensor, xa: torch.Tensor,
                        kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None):
        """
        Override for decoder forward method.
        Parameters:
          x: torch.LongTensor, shape = (batch_size, <= n_ctx) the text tokens
          xa: torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
               the encoded audio features to be attended on
          kv_cache: Dict[str, torch.Tensor], attention modules hidden states cache from previous steps
        """
        if kv_cache is not None:
            offset = kv_cache[0][0].shape[1]
        else:
            offset = 0
            kv_cache = [None for _ in range(len(decoder_.blocks))]
        x = decoder_.token_embedding(
            x) + decoder_.positional_embedding[offset: offset + x.shape[-1]]
        x = x.to(xa.dtype)
        kv_cache_upd = []

        for block, kv_block_cache in zip(decoder_.blocks, kv_cache):
            x, kv_block_cache_upd = block(x, xa, mask=decoder_.mask, kv_cache=kv_block_cache)
            kv_cache_upd.append(tuple(kv_block_cache_upd))

        x = decoder_.ln(x)
        # logits = (x @ torch.transpose(decoder_.token_embedding.weight.to(x.dtype), 1, 0)).float()
        logits = decoder_.linear(x).float()

        return logits, tuple(kv_cache_upd)

    # override decoder forward
    decoder.forward = partial(decoder_forward, decoder)


def convert_pt_encoder_to_ov_through_onnx(encoder, onnx_save_dir):
    mel = torch.zeros((1, 80, 3000))
    encoder(mel)
    torch.onnx.export(
        encoder,
        mel,
        str(onnx_save_dir / "whisper_encoder.onnx"),
        input_names=["mel"],
        output_names=["output_features"]
    )
    ov_encoder = openvino.tools.mo.convert_model(onnx_save_dir / "whisper_encoder.onnx", compress_to_fp16=True)
    return ov_encoder


def convert_pt_encoder_to_ov_directly(encoder, save_dir):
    mel = torch.zeros((1, 80, 3000))
    encoder_model = openvino.tools.ovc.convert_model(encoder, example_input=mel)
    ov.serialize(encoder_model, xml_path=save_dir / "whisper_encoder.xml")


def convert_pt_decoder_to_ov_through_onnx(encoder, decoder, onnx_save_dir, decoder_already_patched):
    if not decoder_already_patched:
        patch_whisper_decoder_for_export(decoder)

    mel = torch.zeros((1, 80, 3000))
    audio_features = encoder(mel)
    tokens = torch.ones((5, 3), dtype=torch.int64)
    _, kv_cache = decoder(tokens, audio_features, kv_cache={})

    kv_cache = {k: v for k, v in kv_cache.items()}
    tokens = torch.ones((5, 1), dtype=torch.int64)
    outputs = [f"out_{k}" for k in kv_cache.keys()]
    inputs = [f"in_{k}" for k in kv_cache.keys()]
    dynamic_axes = {
        "tokens": {0: "beam_size", 1: "seq_len"},
        "audio_features": {0: "beam_size"},
        "logits": {0: "beam_size", 1: "seq_len"}}
    dynamic_outs = {o: {0: "beam_size", 1: "prev_seq_len"} for o in outputs}
    dynamic_inp = {i: {0: "beam_size", 1: "prev_seq_len"} for i in inputs}
    dynamic_axes.update(dynamic_outs)
    dynamic_axes.update(dynamic_inp)

    onnx_save_path = onnx_save_dir / 'whisper_decoder.onnx'
    torch.onnx.export(
        decoder, {'x': tokens, 'xa': audio_features, 'kv_cache': kv_cache},
        onnx_save_path,
        input_names=["tokens", "audio_features"] + inputs,
        output_names=["logits"] + outputs,
        dynamic_axes=dynamic_axes
    )

    input_shapes = "tokens[1..5 1..511],audio_features[1..5 1500 512]"
    for k, v in kv_cache.items():
        if k.endswith('a'):
            input_shapes += f",in_{k}[1..5 0..511 512]"
    ov_decoder = openvino.tools.mo.convert_model(
        input_model=onnx_save_path,
        compress_to_fp16=True,
        input=input_shapes)

    return ov_decoder


def convert_pt_decoder_to_ov_directly(encoder, decoder, save_dir, decoder_already_patched):
    if not decoder_already_patched:
        patch_whisper_decoder_for_export_ovc(decoder)

    tokens = torch.ones((5, 3), dtype=torch.int64)
    mel = torch.zeros((1, 80, 3000))
    audio_features = encoder(mel)
    logits, kv_cache = decoder(tokens, audio_features, kv_cache=None)
    tokens = torch.ones((5, 1), dtype=torch.int64)
    decoder_model = openvino.tools.ovc.convert_model(decoder, example_input=(tokens, audio_features, kv_cache))
    ov.serialize(decoder_model, save_dir / "whisper_decoder.xml")


def filter_decoder_init_data():
    global decoder_init_data

    new_decoder_init_data = []

    n_data = len(decoder_init_data)
    # take last
    filtered_decoder_init_data = []
    seq_key = list(decoder_init_data[0].keys())[2]
    for i in range(len(decoder_init_data) - 2):
        cur_dict, next_dict, next_next_dict = decoder_init_data[i], decoder_init_data[i + 1], decoder_init_data[i + 2]
        next_seq_len = next_dict[seq_key].shape[1]
        next_next_seq_len = next_next_dict[seq_key].shape[1]

        if i == n_data - 3 or (next_next_seq_len == next_seq_len == 0):
            for j in range(5):
                if 0 <= i - j + 2 < n_data:
                    filtered_decoder_init_data.append(decoder_init_data[i - j + 2])
    new_decoder_init_data.extend(filtered_decoder_init_data)

    # take first
    filtered_decoder_init_data = []
    seq_key = list(decoder_init_data[0].keys())[2]
    for i in range(len(decoder_init_data) - 1):
        cur_dict, next_dict = decoder_init_data[i], decoder_init_data[i + 1]
        seq_len, next_seq_len = cur_dict[seq_key].shape[1], next_dict[seq_key].shape[1]
        if seq_len == next_seq_len == 0:
            for j in range(5):
                if i + j < n_data:
                    filtered_decoder_init_data.append(decoder_init_data[i + j])
    new_decoder_init_data.extend(filtered_decoder_init_data)

    decoder_init_data = new_decoder_init_data


def load_init_data(calibration_cache_path, decoder_model_inputs=None):
    global encoder_init_data, decoder_init_data
    with open(calibration_cache_path, 'rb') as f:
        encoder_init_data, decoder_init_data = pickle.load(f)
        encoder_init_data, decoder_init_data = convert_input_data_to_ov_tensor(encoder_init_data), \
            convert_input_data_to_ov_tensor(decoder_init_data)

    if decoder_model_inputs is not None and OVC_API_DECODER and 'tokens' in decoder_init_data[0].keys():
        # calibration data was collected in MO export format and with OVC export input names are different
        input_name_mapping = {'tokens': 'x', 'audio_features': 'xa'}
        new_input_names = [next(iter(inp.names)) for inp in decoder_model_inputs][2:]  # model.decoder.model.inputs
        for new_inp_name, old_inp_name in zip(new_input_names, list(decoder_init_data[0].keys())[2:]):
            input_name_mapping[old_inp_name] = new_inp_name
        for d in decoder_init_data:
            for k in list(d.keys()):
                v = d[k]
                del d[k]
                d[input_name_mapping[k]] = v


def arg_logger(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        logger.info(f"{func} called with args: {args} and kwargs: {kwargs}")
        return func(*args, **kwargs)

    return new_func


def run_benchmark(model_path: Path, shape: str = None, verbose: bool = True) -> float:
    command = f"~/venvs/ov_notebooks/bin/benchmark_app -m {model_path} -d CPU -api async -t 15 -hint latency"
    report_folder = model_path.parent / f'report_{model_path.stem}'
    if not report_folder.exists():
        report_folder.mkdir(parents=True)
    command += f' --report_type average_counters --report_folder={report_folder}'
    command += f' --exec_graph_path={model_path.parent / model_path.stem}_exec_graph.xml'
    if shape is not None:
        command += f' -shape {shape}'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec
    if verbose:
        print(*str(cmd_output).split("\\n")[-9:-1], sep="\n")
    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


def run_validation(model, dataset, return_only_accuracy=True, temperatures=None):
    global num_encoder_forwards, num_decoder_forwards

    processor = WhisperProcessor.from_pretrained("openai/whisper-large")

    total_num_encoder_forwards = []
    total_num_decoder_forwards = []
    total_transcribe_time = 0

    kwargs = {}
    if temperatures is not None:
        kwargs["temperature"] = temperatures

    ground_truths = []
    predictions = []
    for b in tqdm(dataset, disable=return_only_accuracy):
        reference = processor.tokenizer._normalize(b.get("text", b.get("transcription", b.get("sentence"))))

        num_encoder_forwards = num_decoder_forwards = 0

        s_time = datetime.datetime.now()
        transcription = model.transcribe(b["audio"]["array"].astype(np.float32), **kwargs)['text']
        total_transcribe_time += (datetime.datetime.now() - s_time).total_seconds()

        total_num_encoder_forwards.append(num_encoder_forwards)
        total_num_decoder_forwards.append(num_decoder_forwards)

        prediction = processor.tokenizer._normalize(transcription)

        ground_truths.append(reference)
        predictions.append(prediction)

    word_accuracy = 1 - wer.compute(references=ground_truths, predictions=predictions)

    if return_only_accuracy:
        return word_accuracy
    return word_accuracy, ground_truths, predictions, total_transcribe_time, \
        total_num_encoder_forwards, total_num_decoder_forwards


@arg_logger
def quantize(save_dir, encoder_compression, decoder_compression, use_pot, sq_alpha_encoder, sq_alpha_decoder,
             ignore_logits, inplace_statistics_decoder,
             max_encoder_calibration_samples, max_decoder_calibration_samples, num_calibration_samples=30,
             reverse_encoder_calibration_data=False, reverse_decoder_calibration_data=False, decoder_ignored_scope=None,
             filter_init_data=False):
    global encoder_init_data, decoder_init_data

    assert encoder_compression in ["weights", "quantization", None]
    assert decoder_compression in ["weights", "quantization", None]

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    log_file_path = Path(save_dir) / f"quantize_{datetime.datetime.now()}.log"
    map(logger.removeHandler, logger.handlers)
    logger.addHandler(logging.FileHandler(log_file_path))
    nncf_logger.addHandler(logging.FileHandler(log_file_path))
    nncf_logger.setLevel(logging.INFO)

    model = whisper.load_model("base")
    model.to("cpu")
    model.eval()
    original_encoder, original_decoder = model.encoder, model.decoder

    try:
        logger.info(f'Calibration dataset info: {CALIBRATION_DATASET.info}')
    except:
        pass
    if encoder_compression == "quantization" or decoder_compression == "quantization":
        calibration_cache_path = BASE_DIR / CALIBRATION_DATA_CACHE.format(num_calibration_samples)
        if not calibration_cache_path.exists() or not LOAD_CALIBRATION_DATA:
            global COLLECT_INIT_DATA

            patch_whisper_for_ov_inference(model)
            model.encoder = OpenVINOAudioEncoder(core, BASE_DIR / ORIGINAL_ENCODER_MODEL_DIR / 'whisper_encoder.xml')
            model.decoder = OpenVINOTextDecoder(core, BASE_DIR / ORIGINAL_DECODER_MODEL_DIR / 'whisper_decoder.xml')

            # collect model inputs for quantization
            COLLECT_INIT_DATA = True
            logger.info('Collecting calibration data...')
            collection_start_time = datetime.datetime.now()
            # grouped_encoder_init_data, grouped_decoder_init_data = [], []
            for i, data_item in tqdm(enumerate(CALIBRATION_DATASET)):
                if i == num_calibration_samples:
                    break
                model.transcribe(data_item["audio"]["array"].astype(np.float32), beam_size=5, best_of=5, task=TASK)
                # model.transcribe(data_item["audio"]["array"].astype(np.float32), task=TASK)
                # grouped_encoder_init_data.append(encoder_init_data)
                # grouped_decoder_init_data.append(decoder_init_data)
                # encoder_init_data, decoder_init_data = [], []
            # encoder_init_data, decoder_init_data = grouped_encoder_init_data, grouped_decoder_init_data
            # grouped_encoder_init_data = list(reversed(grouped_encoder_init_data))
            # grouped_decoder_init_data = list(reversed(grouped_decoder_init_data))
            # encoder_init_data, decoder_init_data = sum(grouped_encoder_init_data, []), sum(grouped_decoder_init_data, [])

            # model.transcribe(get_audio(video_path), beam_size=5, best_of=5, task=TASK)

            # encoder_init_data, decoder_init_data = list(reversed(encoder_init_data)), list(reversed(decoder_init_data))

            COLLECT_INIT_DATA = False
            logger.info(f'Collecting calibration data took {datetime.datetime.now() - collection_start_time}')

            if SAVE_CALIBRATION_DATA:
                if not calibration_cache_path.parent.exists():
                    calibration_cache_path.parent.mkdir(parents=True)
                with open(calibration_cache_path, 'wb') as f:
                    pickle.dump((convert_input_data_to_np(encoder_init_data),
                                 convert_input_data_to_np(decoder_init_data)), f)
            model.encoder = original_encoder
            model.decoder = original_decoder
        else:
            logger.info('Loading calibration data...')
            ov_decoder = OpenVINOTextDecoder(core, BASE_DIR / ORIGINAL_DECODER_MODEL_DIR / 'whisper_decoder.xml').model
            load_init_data(calibration_cache_path, ov_decoder.inputs)

        if filter_init_data:
            filter_decoder_init_data()

    compressed_model_path = BASE_DIR / save_dir

    #
    # Encoder quantization
    #

    advanced_parameters = AdvancedQuantizationParameters(
        backend_params={"use_pot": use_pot},
        overflow_fix=OverflowFix.DISABLE,
        # disable overflow fix (can lead to accuracy drop on legacy platforms w/o DL Boost),
        smooth_quant_alpha=sq_alpha_encoder
    )
    logger.info(advanced_parameters)

    if encoder_compression == "weights":
        if OVC_API_ENCODER:
            compressed_encoder = nncf.compress_weights(model.encoder, use_fake_quantize=False)
            convert_pt_encoder_to_ov_directly(compressed_encoder, compressed_model_path)
        else:
            compressed_encoder = nncf.compress_weights(model.encoder, use_fake_quantize=True)
            compressed_encoder = convert_pt_encoder_to_ov_through_onnx(compressed_encoder, compressed_model_path)
            quantized_model_path = compressed_model_path / "whisper_encoder.xml"
            ov.serialize(compressed_encoder, str(quantized_model_path))
    else:
        del model.encoder
        model.encoder = OpenVINOAudioEncoder(core, BASE_DIR / ORIGINAL_ENCODER_MODEL_DIR / 'whisper_encoder.xml')

        if encoder_compression == "quantization":
            quantization_dataset = nncf.Dataset(list(reversed(encoder_init_data)) if reverse_encoder_calibration_data
                                                else encoder_init_data)
            if max_encoder_calibration_samples is None:
                max_encoder_calibration_samples = len(encoder_init_data)
            compressed_encoder = nncf.quantize(
                model.encoder.model,
                quantization_dataset,
                model_type=nncf.ModelType.TRANSFORMER,
                fast_bias_correction=True,
                subset_size=min(max_encoder_calibration_samples, len(encoder_init_data)),
                advanced_parameters=advanced_parameters,
                # ignored_scope=ignored_scope,
            )
            del encoder_init_data
            del quantization_dataset
        elif encoder_compression is None:
            compressed_encoder = model.encoder.model
        else:
            raise Exception

        quantized_model_path = compressed_model_path / "whisper_encoder.xml"
        ov.serialize(compressed_encoder, str(quantized_model_path))

    #
    # Decoder quantization
    #

    advanced_parameters = AdvancedQuantizationParameters(
        backend_params={"use_pot": use_pot},
        overflow_fix=OverflowFix.DISABLE,
        # disable overflow fix (can lead to accuracy drop on legacy platforms w/o DL Boost),
        smooth_quant_alpha=sq_alpha_decoder,
        inplace_statistics=inplace_statistics_decoder,
        # activations_range_estimator_params=RangeEstimatorParameters(
        #     min=StatisticsCollectorParameters(
        #         statistics_type=StatisticsType.QUANTILE, aggregator_type=AggregatorType.MIN, quantile_outlier_prob=1e-4
        #     ),
        #     max=StatisticsCollectorParameters(
        #         statistics_type=StatisticsType.QUANTILE, aggregator_type=AggregatorType.MAX, quantile_outlier_prob=1e-4
        #     ),
        # ),
    )
    logger.info(advanced_parameters)

    if decoder_compression == "weights":
        from whisper.model import Linear
        register_module()(Linear)
        model.decoder.linear = Linear(512, 51865, bias=False)
        model.decoder.linear.weight = model.decoder.token_embedding.weight

        if OVC_API_DECODER:
            patch_whisper_decoder_for_export_ovc(model.decoder)
            compressed_decoder = nncf.compress_weights(model.decoder, use_fake_quantize=False)
            # compressed_decoder = model.decoder
            # logger.info(compressed_decoder)
            convert_pt_decoder_to_ov_directly(original_encoder, compressed_decoder, compressed_model_path,
                                              decoder_already_patched=True)
        else:
            patch_whisper_decoder_for_export(model.decoder)
            compressed_decoder = nncf.compress_weights(model.decoder, use_fake_quantize=True)
            compressed_decoder = convert_pt_decoder_to_ov_through_onnx(original_encoder, compressed_decoder,
                                                                       compressed_model_path,
                                                                       decoder_already_patched=True)
            quantized_model_path = compressed_model_path / "whisper_decoder.xml"
            ov.serialize(compressed_decoder, str(quantized_model_path))
    else:
        patch_whisper_for_ov_inference(model)
        del model.decoder
        model.decoder = OpenVINOTextDecoder(core, BASE_DIR / ORIGINAL_DECODER_MODEL_DIR / 'whisper_decoder.xml')

        if decoder_compression == "quantization":
            quantization_dataset = nncf.Dataset(list(reversed(decoder_init_data)) if reverse_decoder_calibration_data
                                                else decoder_init_data)
            if decoder_ignored_scope is None and ignore_logits:
                decoder_ignored_scope = IgnoredScope(names=["aten::to/Convert_713" if OVC_API_DECODER else "logits"])
            if max_decoder_calibration_samples is None:
                max_decoder_calibration_samples = len(decoder_init_data)
            compressed_decoder = nncf.quantize(
                model.decoder.model,
                quantization_dataset,
                model_type=nncf.ModelType.TRANSFORMER,
                fast_bias_correction=True,
                subset_size=min(max_decoder_calibration_samples, len(decoder_init_data)),
                advanced_parameters=advanced_parameters,
                ignored_scope=decoder_ignored_scope
            )
            del decoder_init_data
            del quantization_dataset
        elif decoder_compression is None:
            compressed_decoder = model.decoder.model
        else:
            raise Exception

        quantized_model_path = compressed_model_path / "whisper_decoder.xml"
        ov.serialize(compressed_decoder, str(quantized_model_path))

    return compressed_model_path


@arg_logger
def benchmark(model_path):
    log_file_path = model_path / f"benchmark_{datetime.datetime.now()}.log"
    map(logger.removeHandler, logger.handlers)
    logger.addHandler(logging.FileHandler(log_file_path))

    model = whisper.load_model("base")
    model.to("cpu")
    model.eval()
    patch_whisper_for_ov_inference(model)
    model.encoder = OpenVINOAudioEncoder(core, model_path / "whisper_encoder.xml")
    model.decoder = OpenVINOTextDecoder(core, model_path / "whisper_decoder.xml")

    if OVC_API_DECODER:
        decoder_shape = 'x[5,1],xa[5,1500,512],'
    else:
        decoder_shape = 'tokens[5,1],audio_features[5,1500,512],'

    if OVC_API_ENCODER:
        encoder_shape = 'x[1,80,3000]'
    else:
        encoder_shape = None

    input_names = [next(iter(inp.names)) for inp in model.decoder.model.inputs][2:]
    decoder_shape += ','.join([f"{name}[5,1,512]" for name in input_names])

    logger.info(f'Benchmarking model {model_path}')
    encoder_fps = run_benchmark(Path(model_path) / "whisper_encoder.xml", verbose=True, shape=encoder_shape)
    decoder_fps = run_benchmark(Path(model_path) / "whisper_decoder.xml", verbose=True, shape=decoder_shape)
    logger.info(f'Encoder FPS: {encoder_fps}')
    logger.info(f'Decoder FPS: {decoder_fps}')


@arg_logger
def transcribe_video(model_path, video_path, temperatures=None):
    global start_time, VERBOSE, video_transcription_ground_truths, num_encoder_forwards, num_decoder_forwards

    verbose_value = VERBOSE
    # verbose = bool(1)

    log_file_path = model_path / f"transcribe_{datetime.datetime.now()}.log"
    map(logger.removeHandler, logger.handlers)
    logger.addHandler(logging.FileHandler(log_file_path))

    logger.info(f'Transcribing model from path {model_path}')

    model = whisper.load_model("base")
    model.to("cpu")
    model.eval()
    patch_whisper_for_ov_inference(model)
    model.encoder = OpenVINOAudioEncoder(core, model_path / "whisper_encoder.xml")
    model.decoder = OpenVINOTextDecoder(core, model_path / "whisper_decoder.xml")

    audio = get_audio(video_path)

    num_encoder_forwards, num_decoder_forwards = 0, 0

    start_time = datetime.datetime.now()
    kwargs = dict()
    if temperatures is not None:
        kwargs["temperature"] = temperatures
    transcription = model.transcribe(audio, beam_size=5, best_of=5, task=TASK, **kwargs)
    finish_time = datetime.datetime.now()

    logger.info(f'Transcription: {transcription["text"]}')
    for gt in video_transcription_ground_truths:
        word_accuracy = 1 - wer.compute(references=[transcription["text"]], predictions=[gt])
        measures = compute_measures(transcription["text"], gt)
        S, D, I, H = measures["substitutions"], measures["deletions"], measures["insertions"], measures["hits"]
        logger.info(f"Accuracy: {word_accuracy:.04f}; S: {S}; D: {D}; I: {I}; H: {H}")

    srt_lines = prepare_srt(transcription)

    logger.info("".join(srt_lines))

    logger.info(f'Number of encoder calls: {num_encoder_forwards}')
    logger.info(f'Number of decoder calls: {num_decoder_forwards}')
    logger.info(f'Transcribing time: {finish_time - start_time}')

    VERBOSE = verbose_value


@arg_logger
def validate_model(model_path, temperatures=None):
    log_file_path = model_path / f"validation_{datetime.datetime.now()}.log"
    map(logger.removeHandler, logger.handlers)
    logger.addHandler(logging.FileHandler(log_file_path))

    logger.info(f'Validating model {model_path} with temperatures: {temperatures}')
    model = whisper.load_model("base")
    model.to("cpu")
    model.eval()
    patch_whisper_for_ov_inference(model)
    model.encoder = OpenVINOAudioEncoder(core, model_path / "whisper_encoder.xml")
    model.decoder = OpenVINOTextDecoder(core, model_path / "whisper_decoder.xml")

    dataset = load_dataset("librispeech_asr", "clean", split="test[:100]")
    # dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="test", streaming=True).take(100)
    # dataset = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")

    logger.info(f'Dataset info: {dataset.info}')

    word_accuracy, ground_truths, predictions, total_transcribe_time, \
        total_num_encoder_forwards, total_num_decoder_forwards = \
        run_validation(model, list(dataset), return_only_accuracy=False, temperatures=temperatures)

    total_S, total_D, total_I, total_H = 0, 0, 0, 0
    for ref, pred in zip(ground_truths, predictions):
        measures = compute_measures(ref, pred)
        S, D, I, H = measures["substitutions"], measures["deletions"], measures["insertions"], measures["hits"]
        total_S += S;
        total_D += D;
        total_I += I;
        total_H += H
        wer = (S + D + I) / (S + D + H)
        logger.info(f"WER: {wer:.04f}; S: {S}; D: {D}; I: {I}; H: {H}")
        logger.info(f"Reference : {ref}")
        logger.info(f"Prediction: {pred}")
        logger.info("\n")

    logger.info(f'Word accuracy (1-wer): {word_accuracy}')
    logger.info(f'Total transcribe time: {total_transcribe_time}', )
    logger.info(f'Average encoder calls: {sum(total_num_encoder_forwards) / len(total_num_encoder_forwards)}')
    logger.info(f'Average decoder calls: {sum(total_num_decoder_forwards) / len(total_num_decoder_forwards)}')
    logger.info(f"Total S: {total_S}; D: {total_D}; I: {total_I}; H: {total_H}")


@arg_logger
def quantize_decoder_with_accuracy_control(model_path, save_dir, sq_alpha_decoder, max_drop, temperatures=None):
    global decoder_init_data

    model_path = Path(model_path)
    save_dir = model_path / save_dir

    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    log_file_path = Path(save_dir) / f"qwac_{datetime.datetime.now()}.log"
    map(logger.removeHandler, logger.handlers)
    logger.addHandler(logging.FileHandler(log_file_path))
    nncf_logger.addHandler(logging.FileHandler(log_file_path))

    model = whisper.load_model("base")
    model.to("cpu")
    model.eval()
    patch_whisper_for_ov_inference(model)
    model.encoder = OpenVINOAudioEncoder(core, model_path / "whisper_encoder.xml")
    model.decoder = None
    ov_decoder = core.read_model(model_path / "whisper_decoder.xml")

    logger.info('Loading calibration data...')
    load_init_data(BASE_DIR / CALIBRATION_DATA_CACHE.format(30), ov_decoder.inputs)
    quantization_dataset = nncf.Dataset(list(reversed(decoder_init_data)))

    ptq_config = dict(
        advanced_quantization_parameters=AdvancedQuantizationParameters(
            overflow_fix=OverflowFix.DISABLE,
            smooth_quant_alpha=sq_alpha_decoder
        ),
        model_type=nncf.ModelType.TRANSFORMER,
        fast_bias_correction=True,
        subset_size=min(300, len(decoder_init_data)),
        max_drop=max_drop,
    )

    dataset = load_dataset("librispeech_asr", "clean", split="test[:100]")
    # dataset = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")

    def validation_fn(m, d):
        model.decoder = OpenVINOTextDecoder(core, m)
        return run_validation(model, d, return_only_accuracy=True, temperatures=temperatures)

    ov.serialize(ov_decoder, save_dir / "whisper_decoder_initial.xml")
    compressed_decoder = nncf.quantize_with_accuracy_control(
        ov_decoder,
        quantization_dataset,
        nncf.Dataset(list(dataset)),
        validation_fn,
        **ptq_config
    )

    ov.serialize(compressed_decoder, save_dir / "whisper_decoder.xml")
    ov.serialize(model.encoder.model, save_dir / "whisper_encoder.xml")


video_path, video_transcription_ground_truths = (
    download_video(BASE_DIR, "https://youtu.be/kgL5LBM-hFI"),
    ["Oh, what's that? Oh, wow. Hello, humans. Focus on me. Focus on the guard. Don't tell anyone what you've seen in "
     "here.",
     "Oh, what's that? Oh, wow. Hello, humans. Focus on me. Focus on the guard. Don't tell anyone what you've seen in "
     "here. Have you seen what's in there? They have. Intel, this is where it all changes."])  # Intel 1
# video_path = download_video(base_dir, "https://www.youtube.com/watch?v=zasAa3Wgdp0")  # Intel 2
# video_path = download_video(base_dir, "https://www.youtube.com/watch?v=JzPfMbG1vrE")  # Other 30 sec
# video_path = download_video(base_dir, "https://www.youtube.com/watch?v=I8iBhUMFCIA")  # Other 45 sec


compressed_model_path = quantize("calibration_datasets/librispeech_asr_train100_clean/transcribe/15_att3",
                                 # encoder_compression="weights",
                                 encoder_compression="quantization",
                                 # encoder_compression=None,
                                 # decoder_compression="weights",
                                 decoder_compression="quantization",
                                 # decoder_compression=None,
                                 use_pot=bool(0),
                                 sq_alpha_encoder=0.50,
                                 sq_alpha_decoder=0.95,
                                 ignore_logits=bool(0),
                                 inplace_statistics_decoder=bool(1),
                                 num_calibration_samples=15,
                                 max_encoder_calibration_samples=None,
                                 max_decoder_calibration_samples=None,
                                 reverse_encoder_calibration_data=bool(0),
                                 reverse_decoder_calibration_data=bool(0),
                                 filter_init_data=bool(0),
                                 # decoder_ignored_scope=ignored_scope3
                                 )
# benchmark(compressed_model_path)
transcribe_video(compressed_model_path, video_path)
# validate_model(compressed_model_path)

# model_path = Path("./int8")
# benchmark(model_path)
# transcribe_video(model_path, video_path, temperatures=None)
# validate_model(model_path, temperatures=None)


# quantize_decoder_with_accuracy_control("quantized_models/ovc/encoder-ptq-sq-0.50_decoder-none",
#                                        save_dir="qwac_0.005_att3",
#                                        sq_alpha_decoder=0.95,
#                                        max_drop=0.005,
#                                        temperatures=None)


if OVC_API_ENCODER:
    logger.info('Used OVC API with conversion straight to IR for encoder')
if OVC_API_DECODER:
    logger.info('Used OVC API with conversion straight to IR for decoder')
