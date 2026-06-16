"""Helper for the OmniVoice OpenVINO notebook.

Provides:

- ``convert_omnivoice``: convert the OmniVoice LLM, the HiggsAudio v2
  audio tokenizer (encoder + decoder), and (optionally) Whisper-large-v3-turbo
  to OpenVINO IR.

- ``OVOmniVoice``: an inference wrapper that mirrors the original
  ``omnivoice.OmniVoice`` API (``generate``, ``create_voice_clone_prompt``,
  ``transcribe``, ``sampling_rate``, ``text_tokenizer``) but runs the
  three weighted sub-models on OpenVINO. The 32-step diffusion loop,
  classifier-free guidance, sampling and pre/post-processing live in
  Python and reuse the upstream ``omnivoice`` package helpers.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Union

import numpy as np
import openvino as ov

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


LLM_FILE = "openvino_llm_model.xml"
ENCODER_FILE = "openvino_audio_encoder.xml"
DECODER_FILE = "openvino_audio_decoder.xml"
WHISPER_DIR = "whisper"


def _bool_to_additive_mask(attention_mask, dtype):
    """Convert a [B,1,S,S] bool mask (True=keep) to an additive float mask
    (0 where True, large-negative where False), as expected by Qwen3 eager
    attention."""
    import torch

    # Use a finite negative number (instead of -inf) so OpenVINO conversion
    # is happy and softmax stays well-defined.
    neg = torch.finfo(dtype).min
    return (~attention_mask).to(dtype) * neg


class _OmniVoiceLLMWrapper:
    """nn.Module that exposes the OmniVoice forward as a single graph
    suitable for ``ov.convert_model``.

    Inputs:
        input_ids: ``[B, num_codebook, S]`` (long)
        audio_mask: ``[B, S]`` (bool)
        attention_mask: ``[B, 1, S, S]`` (bool, True = keep)

    Output:
        logits: ``[B, num_codebook, S, audio_vocab_size]`` (float)
    """

    def __new__(cls, model):
        import torch
        from torch import nn

        class _Inner(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, audio_mask, attention_mask):
                # Mixed text+audio embedding (matches OmniVoice._prepare_embed_inputs).
                m = self.m
                text_embeds = m.get_input_embeddings()(input_ids[:, 0, :])
                shifted_ids = (input_ids * audio_mask.unsqueeze(1).long()) + m.codebook_layer_offsets.view(1, -1, 1)
                audio_embeds = m.audio_embeddings(shifted_ids).sum(dim=1)
                inputs_embeds = torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

                # Convert the bool 4D mask to additive form so eager attention
                # can simply add it to the logits.
                add_mask = _bool_to_additive_mask(attention_mask, inputs_embeds.dtype)

                llm_out = m.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=add_mask,
                    use_cache=False,
                    return_dict=True,
                )
                hidden_states = llm_out[0]

                bsz, seq_len, _ = hidden_states.shape
                logits_flat = m.audio_heads(hidden_states)
                # [B, S, C, V] -> [B, C, S, V]
                audio_logits = logits_flat.view(
                    bsz,
                    seq_len,
                    m.config.num_audio_codebook,
                    m.config.audio_vocab_size,
                ).permute(0, 2, 1, 3)
                return audio_logits

        return _Inner(model)


class _AudioEncoderWrapper:
    """Wraps the HiggsAudio v2 encoder so it can be traced as a single graph.

    The original ``encode`` does a 24k→16k ``torchaudio.functional.resample``
    inside, plus an ``if/else`` based on shape comparison between acoustic and
    semantic feature lengths. Both bake fixed-shape constants under
    ``torch.jit.trace`` and break for input lengths different from the trace
    example. We solve both problems by:

    1. Doing the resample (and the constant-pad of 160 samples each side) in
       Python BEFORE calling the OV graph, so the graph receives both the
       24k input (for the acoustic branch) and the 16k+padded input (for the
       semantic branch).
    2. Always taking the no-pad branch (``input_length`` of acoustic equals
       semantic frame count). Per inspection on every length we tested,
       ``need_pad`` is always False; we hard-code that.

    Inputs:
        - ``input_values_24k``: ``[1, 1, T_24k]`` waveform at 24kHz, T multiple
          of ``hop_length`` (960).
        - ``input_values_16k_padded``: ``[1, T_16k_padded]`` waveform at 16kHz
          padded ``(160, 160)`` on each side.

    Output: ``[1, 8, T_24k/hop]`` codes (matches the public ``encode`` API
    after the internal ``transpose(0, 1)``).
    """

    def __new__(cls, audio_tokenizer):
        import torch
        import torch.nn.functional as F
        from torch import nn

        class _Inner(nn.Module):
            def __init__(self, t):
                super().__init__()
                self.t = t

            def forward(self, input_values_24k, input_values_16k_padded):
                t = self.t

                # --- Semantic branch (HuBERT + SemanticEncoder) ---
                with torch.no_grad():
                    sem_out = t.semantic_model(input_values_16k_padded, output_hidden_states=True)
                hidden_states = sem_out.hidden_states
                stacked = torch.stack(list(hidden_states), dim=1)
                semantic_features = stacked.mean(dim=1)
                if t.config.semantic_downsample_factor > 1:
                    semantic_features = semantic_features[:, :: t.config.semantic_downsample_factor, :]
                e_semantic = t.encoder_semantic(semantic_features.transpose(1, 2))

                # --- Acoustic branch ---
                # Always take the no-pad path: at every tested length the
                # acoustic conv output length equals the semantic frame count.
                e_acoustic = t.acoustic_encoder(input_values_24k)

                # --- Concat + project + quantize ---
                embeddings = torch.cat([e_acoustic, e_semantic], dim=1)
                embeddings = t.fc(embeddings.transpose(1, 2)).transpose(1, 2)
                bw = t.config.target_bandwidths[-1]
                audio_codes = t.quantizer.encode(embeddings, bw)
                # Original `encode` then transposes to `(B, C, T)`
                audio_codes = audio_codes.transpose(0, 1)
                return audio_codes

        return _Inner(audio_tokenizer)


class _AudioDecoderWrapper:
    """Wraps the HiggsAudio v2 decoder. Input: ``[8, 1, N]`` codes.
    Output: ``[1, 1, N*960]`` waveform."""

    def __new__(cls, audio_tokenizer):
        from torch import nn

        class _Inner(nn.Module):
            def __init__(self, t):
                super().__init__()
                self.t = t

            def forward(self, audio_codes):
                return self.t.decode(audio_codes).audio_values

        return _Inner(audio_tokenizer)


def _save_ov(model, path: Path, compress_to_fp16: bool = True):
    ov.save_model(model, str(path), compress_to_fp16=compress_to_fp16)


def _convert_llm(model, output_path: Path, quantization_config: Optional[dict]):
    import torch
    from openvino.frontend.pytorch.patch_model import __make_16bit_traceable

    print("⌛ Convert OmniVoice LLM (Qwen3 + audio embed + audio_heads)")

    # Force eager attention so create_causal_mask is bypassed (we feed a 4D mask).
    model.llm.config._attn_implementation = "eager"
    model.llm.config.use_cache = False
    if hasattr(model.llm, "model"):
        model.llm.model.config._attn_implementation = "eager"
        model.llm.model.config.use_cache = False

    wrapper = _OmniVoiceLLMWrapper(model)
    wrapper.eval()
    __make_16bit_traceable(wrapper)

    B, C, S = 2, model.config.num_audio_codebook, 16
    example_input_ids = torch.zeros((B, C, S), dtype=torch.long)
    example_audio_mask = torch.zeros((B, S), dtype=torch.bool)
    example_audio_mask[:, S // 2 :] = True
    example_attn = torch.ones((B, 1, S, S), dtype=torch.bool)

    with torch.no_grad():
        ov_model = ov.convert_model(
            wrapper,
            example_input=(example_input_ids, example_audio_mask, example_attn),
            input=[
                ov.PartialShape([-1, C, -1]),
                ov.PartialShape([-1, -1]),
                ov.PartialShape([-1, 1, -1, -1]),
            ],
        )
    ov_model.inputs[0].get_node().set_friendly_name("input_ids")
    ov_model.inputs[1].get_node().set_friendly_name("audio_mask")
    ov_model.inputs[2].get_node().set_friendly_name("attention_mask")

    if quantization_config is not None:
        import nncf

        print("⌛ Compress LLM weights with NNCF (INT8)")
        ov_model = nncf.compress_weights(ov_model, **quantization_config)

    _save_ov(ov_model, output_path, compress_to_fp16=quantization_config is None)
    del ov_model
    gc.collect()
    print("✅ LLM saved to", output_path)


def _convert_audio_tokenizer(
    audio_tokenizer,
    output_dir: Path,
    convert_encoder: bool = True,
    convert_decoder: bool = True,
):
    import torch
    from openvino.frontend.pytorch.patch_model import __make_16bit_traceable

    # Patch HuBERT encoder to skip mask creation (we don't use padding).
    # Without this, masking_utils calls into sdpa_mask which triggers an
    # IndexError under torch.jit.trace on transformers>=5.x.
    if hasattr(audio_tokenizer, "semantic_model"):
        sem = audio_tokenizer.semantic_model
        sem.config._attn_implementation = "eager"

        encoder = sem.encoder

        def _patched_encoder_forward(
            self_,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ):
            from transformers.modeling_outputs import BaseModelOutput

            position_embeddings = self_.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings.to(hidden_states.device)
            hidden_states = self_.layer_norm(hidden_states)
            hidden_states = self_.dropout(hidden_states)
            all_hidden = () if output_hidden_states else None
            for layer in self_.layers:
                if output_hidden_states:
                    all_hidden = all_hidden + (hidden_states,)
                hidden_states = layer(hidden_states, attention_mask=None)[0]
            if output_hidden_states:
                all_hidden = all_hidden + (hidden_states,)
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden,
            )

        import types

        encoder.forward = types.MethodType(_patched_encoder_forward, encoder)

    hop = audio_tokenizer.config.hop_length
    T = (24000 // hop) * hop
    C = audio_tokenizer.config.num_quantizers

    if convert_encoder:
        print("⌛ Convert HiggsAudio encoder")
        enc = _AudioEncoderWrapper(audio_tokenizer)
        enc.eval()
        __make_16bit_traceable(enc)
        # 24k waveform: T_24k samples, multiple of hop (960).
        example_wave_24k = torch.randn((1, 1, T), dtype=torch.float32)
        # 16k waveform pre-resampled in Python and padded (160, 160) on each side.
        T_16k = T * 16000 // 24000  # exact integer for hop-aligned T
        example_wave_16k_padded = torch.zeros((1, T_16k + 320), dtype=torch.float32)
        with torch.no_grad():
            ov_enc = ov.convert_model(
                enc,
                example_input=(example_wave_24k, example_wave_16k_padded),
                input=[
                    ov.PartialShape([1, 1, -1]),
                    ov.PartialShape([1, -1]),
                ],
            )
        ov_enc.inputs[0].get_node().set_friendly_name("input_values_24k")
        ov_enc.inputs[1].get_node().set_friendly_name("input_values_16k_padded")
        _save_ov(ov_enc, output_dir / ENCODER_FILE)
        del ov_enc, enc
        gc.collect()
        print("✅ Audio encoder saved")

    if convert_decoder:
        print("⌛ Convert HiggsAudio decoder")
        dec = _AudioDecoderWrapper(audio_tokenizer)
        dec.eval()
        __make_16bit_traceable(dec)
        # HiggsAudio.decode takes (B, C, T) and transposes internally; we trace
        # with this shape so the OV signature matches the original API.
        example_codes = torch.zeros((1, C, T // hop), dtype=torch.long)
        with torch.no_grad():
            ov_dec = ov.convert_model(dec, example_input=(example_codes,), input=[ov.PartialShape([1, C, -1])])
        ov_dec.inputs[0].get_node().set_friendly_name("audio_codes")
        _save_ov(ov_dec, output_dir / DECODER_FILE)
        del ov_dec, dec
        gc.collect()
        print("✅ Audio decoder saved")


def _convert_whisper(asr_model_id: str, output_dir: Path):
    """Fetch a pre-converted OpenVINO Whisper model.

    We use Intel's pre-converted INT8 Whisper IR (e.g.
    ``OpenVINO/whisper-large-v3-turbo-int8-ov``) instead of converting from
    PyTorch. This avoids the transformers>=5 ``create_causal_mask`` /
    ``sdpa_mask`` tracing bug and gives us a stateful decoder + bundled
    OpenVINO tokenizer/detokenizer that ``openvino_genai.WhisperPipeline``
    consumes natively.

    Mapping:
        ``openai/whisper-large-v3-turbo`` -> ``OpenVINO/whisper-large-v3-turbo-int8-ov``
        ``openai/whisper-medium``         -> ``OpenVINO/whisper-medium-fp16-ov``
        ...

    Pass ``asr_model_id`` already in the ``OpenVINO/...`` form to use it
    verbatim.
    """
    target = output_dir / WHISPER_DIR
    if (target / "openvino_encoder_model.xml").exists():
        print("✓ Whisper already converted")
        return

    from huggingface_hub import snapshot_download

    # If the user passed an openai/* id, swap in the equivalent INT8 OV repo.
    ov_repo_id = _resolve_ov_whisper_id(asr_model_id)

    target.mkdir(parents=True, exist_ok=True)
    print(f"⌛ Downloading pre-converted OpenVINO Whisper: {ov_repo_id}")
    snapshot_download(ov_repo_id, local_dir=str(target))
    print("✅ OV Whisper saved to", target)


_OPENAI_TO_OV_WHISPER = {
    "openai/whisper-large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "openai/whisper-large-v3": "OpenVINO/whisper-large-v3-int8-ov",
    "openai/whisper-medium": "OpenVINO/whisper-medium-fp16-ov",
    "openai/whisper-small": "OpenVINO/whisper-small-fp16-ov",
    "openai/whisper-base": "OpenVINO/whisper-base-fp16-ov",
    "openai/whisper-tiny": "OpenVINO/whisper-tiny-fp16-ov",
}


def _resolve_ov_whisper_id(asr_model_id: str) -> str:
    """Map openai/* repo ids to OpenVINO/* equivalents. Pass-through for
    ids that are already OpenVINO/* or local paths."""
    if asr_model_id.startswith("OpenVINO/") or os.path.isdir(asr_model_id):
        return asr_model_id
    return _OPENAI_TO_OV_WHISPER.get(asr_model_id, asr_model_id)


def convert_omnivoice(
    model_id: str = "k2-fsa/OmniVoice",
    output_dir: Union[str, Path] = "ov_model",
    pt_cache_dir: Union[str, Path, None] = None,
    llm_quantization_config: Optional[dict] = None,
    convert_whisper: bool = True,
    asr_model_id: str = "OpenVINO/whisper-large-v3-turbo-int8-ov",
    local_model_dir: Optional[Union[str, Path]] = None,
):
    """Convert OmniVoice (LLM + audio tokenizer + Whisper) to OpenVINO IR.

    Args:
        model_id: HF repo id of the OmniVoice model.
        output_dir: Where to write the OpenVINO IR + small runtime assets.
            This is the only directory the inference runtime needs.
        pt_cache_dir: Where to cache the original PyTorch checkpoint while
            tracing. Kept separate from ``output_dir`` so the OV folder is
            self-contained. Default: ``./pt_models/<model_id_basename>``.
        llm_quantization_config: kwargs forwarded to ``nncf.compress_weights``.
        convert_whisper: Convert Whisper for ref-text auto-transcription.
        asr_model_id: HF repo id of the Whisper variant.
        local_model_dir: Optional pre-downloaded checkpoint directory. If
            given, files are copied (or symlinked) into ``pt_cache_dir``.
    """
    import torch
    from huggingface_hub import snapshot_download
    from omnivoice import OmniVoice

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve PyTorch checkpoint dir (separate from OV output_dir).
    if pt_cache_dir is None:
        pt_cache_dir = Path("pt_models") / model_id.split("/")[-1]
    pt_cache_dir = Path(pt_cache_dir)

    if local_model_dir is not None and Path(local_model_dir).exists():
        if not pt_cache_dir.exists():
            print(f"⌛ Copying local checkpoint from {local_model_dir}")
            pt_cache_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(local_model_dir, pt_cache_dir)
        resolved = str(pt_cache_dir)
    else:
        if not pt_cache_dir.exists():
            print(f"⌛ Downloading {model_id} -> {pt_cache_dir}")
            snapshot_download(model_id, local_dir=str(pt_cache_dir))
        resolved = str(pt_cache_dir)

    needs_llm = not (output_dir / LLM_FILE).exists()
    needs_encoder = not (output_dir / ENCODER_FILE).exists()
    needs_decoder = not (output_dir / DECODER_FILE).exists()
    needs_whisper = convert_whisper and not (output_dir / WHISPER_DIR / "openvino_encoder_model.xml").exists()

    if needs_llm or needs_encoder or needs_decoder:
        print("⌛ Loading PyTorch OmniVoice on CPU ...")
        model = OmniVoice.from_pretrained(resolved, dtype=torch.float32, device_map="cpu")
        model.eval()

        if needs_llm:
            _convert_llm(model, output_dir / LLM_FILE, llm_quantization_config)
        if needs_encoder or needs_decoder:
            _convert_audio_tokenizer(
                model.audio_tokenizer,
                output_dir,
                convert_encoder=needs_encoder,
                convert_decoder=needs_decoder,
            )

        del model
        gc.collect()
    else:
        print("✓ LLM and audio tokenizer already converted")

    if needs_whisper:
        _convert_whisper(asr_model_id, output_dir)
    elif convert_whisper:
        print("✓ Whisper already converted")

    _copy_runtime_assets(Path(resolved), output_dir)

    print("✅ Conversion complete. OV output:", output_dir)
    print("   PyTorch checkpoint cache (safe to delete after conversion):", pt_cache_dir)


def _copy_runtime_assets(ckpt_dir: Path, output_dir: Path):
    """Copy the small JSON / tokenizer files needed at inference time from the
    PyTorch ckpt into the OV output dir. After this, the runtime no longer
    depends on ``ckpt/`` and the ckpt dir may be deleted."""
    files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    for name in files:
        src = ckpt_dir / name
        if src.exists():
            dst = output_dir / name
            if not dst.exists():
                shutil.copy2(src, dst)

    audio_files = ["config.json", "preprocessor_config.json"]
    src_audio = ckpt_dir / "audio_tokenizer"
    if src_audio.exists():
        dst_audio = output_dir / "audio_tokenizer"
        dst_audio.mkdir(parents=True, exist_ok=True)
        for name in audio_files:
            src = src_audio / name
            if src.exists():
                dst = dst_audio / name
                if not dst.exists():
                    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Inference runtime
# ---------------------------------------------------------------------------


class _OVTorchLLM:
    """Thin shim that exposes ``__call__(input_ids, audio_mask, attention_mask)``
    returning an object with a ``.logits`` attribute, so we can plug into the
    upstream ``OmniVoice._generate_iterative`` loop with minimal changes."""

    def __init__(self, compiled, device):
        self.request = compiled.create_infer_request()
        self.device = device

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    def __call__(self, input_ids, audio_mask, attention_mask):
        import torch

        inputs = {
            "input_ids": input_ids.cpu().numpy().astype(np.int64),
            "audio_mask": audio_mask.cpu().numpy().astype(np.bool_),
            "attention_mask": attention_mask.cpu().numpy().astype(np.bool_),
        }
        result = self.request.infer(inputs)
        logits = next(iter(result.values()))
        return self._Out(torch.from_numpy(logits))


class _OVAudioTokenizerEncode:
    """Returns an object with ``.audio_codes`` like the HF tokenizer.

    The OV graph expects two inputs: 24kHz waveform and 16kHz pre-resampled +
    padded waveform. We do the resample in Python here (using
    ``torchaudio.functional.resample``, the same kernel the original PyTorch
    ``_extract_semantic_features`` uses).
    """

    class _Out:
        def __init__(self, audio_codes):
            self.audio_codes = audio_codes

    def __init__(self, compiled, sample_rate: int = 24000, semantic_sample_rate: int = 16000):
        self.request = compiled.create_infer_request()
        self.sample_rate = sample_rate
        self.semantic_sample_rate = semantic_sample_rate

    def __call__(self, input_values):
        import torch
        import torch.nn.functional as F
        import torchaudio.functional as taF

        # input_values: [1, 1, T_24k]
        wav_24k = input_values
        # Resample to 16k for the semantic branch and pad 160 each side
        # (matches HiggsAudioV2TokenizerModel._extract_semantic_features).
        wav_16k = taF.resample(
            wav_24k[:, 0, :],
            orig_freq=self.sample_rate,
            new_freq=self.semantic_sample_rate,
        )
        wav_16k = F.pad(wav_16k, (160, 160))

        result = self.request.infer(
            {
                "input_values_24k": wav_24k.cpu().numpy(),
                "input_values_16k_padded": wav_16k.cpu().numpy(),
            }
        )
        return self._Out(torch.from_numpy(next(iter(result.values()))).long())


class _OVAudioTokenizerDecode:
    class _Out:
        def __init__(self, audio_values):
            self.audio_values = audio_values

    def __init__(self, compiled):
        self.request = compiled.create_infer_request()

    def __call__(self, audio_codes):
        import torch

        result = self.request.infer({"audio_codes": audio_codes.cpu().numpy().astype(np.int64)})
        return self._Out(torch.from_numpy(next(iter(result.values()))).float())


class _OVAudioTokenizer:
    """Mimics the subset of HiggsAudioV2TokenizerModel used by OmniVoice
    (``encode`` + ``decode`` + ``config`` + ``device``)."""

    def __init__(self, encoder_compiled, decoder_compiled, config):
        self._enc = _OVAudioTokenizerEncode(
            encoder_compiled,
            sample_rate=config.sample_rate,
            semantic_sample_rate=config.semantic_sample_rate,
        )
        self._dec = _OVAudioTokenizerDecode(decoder_compiled)
        self.config = config
        # The upstream code calls .to(self.audio_tokenizer.device); we keep CPU
        # and strip device in the encode/decode shims.
        import torch

        self.device = torch.device("cpu")

    def encode(self, input_values, **_kw):
        return self._enc(input_values)

    def decode(self, audio_codes, **_kw):
        return self._dec(audio_codes)


class _OVWhisperASR:
    """OpenVINO Whisper inference, wrapping ``openvino_genai.WhisperPipeline``.

    Callable like a HuggingFace ``pipeline("automatic-speech-recognition", ...)``
    so it drops into ``OmniVoice.transcribe`` unchanged. Returns a dict with a
    ``"text"`` key.
    """

    _WHISPER_SR = 16000

    def __init__(self, whisper_dir: Path, device: str = "CPU"):
        import openvino_genai

        self._pipe = openvino_genai.WhisperPipeline(str(Path(whisper_dir)), device)

    def __call__(self, audio):
        """Accepts a file path or ``{"array": np.ndarray, "sampling_rate": int}``.
        Resamples to 16kHz and returns ``{"text": <str>}``.
        """
        if isinstance(audio, str):
            import soundfile as sf

            wav, sr = sf.read(audio, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
        else:
            wav = np.asarray(audio["array"], dtype=np.float32).reshape(-1)
            sr = int(audio["sampling_rate"])

        if sr != self._WHISPER_SR:
            import torch
            import torchaudio.functional as taF

            wav = taF.resample(
                torch.from_numpy(wav.astype(np.float32)),
                orig_freq=sr,
                new_freq=self._WHISPER_SR,
            ).numpy()

        result = self._pipe.generate(wav.astype(np.float32))
        # WhisperPipeline returns a DecodedResults object; .texts is a list.
        texts = getattr(result, "texts", None)
        if texts:
            return {"text": str(texts[0])}
        return {"text": str(result)}


class _OmniVoiceStandIn(SimpleNamespace):
    """Lightweight drop-in for the OmniVoice nn.Module at inference time.

    Holds only the small set of attributes the rebound generation methods need
    (config, tokenizers, duration estimator, sampling rate, ASR pipe). The
    LLM forward is dispatched to OpenVINO via ``__call__`` which is overridden
    here so ``self(input_ids=..., audio_mask=..., attention_mask=...)`` works
    inside the upstream ``_generate_iterative`` loop.
    """

    def __call__(self, **kwargs):
        return self._ov_call(**kwargs)


class OVOmniVoice:
    """OpenVINO-backed drop-in for ``omnivoice.OmniVoice``.

    Initialization loads only OpenVINO IR + small JSON / tokenizer assets from
    the OV output dir. No PyTorch model weights are allocated; the original
    ``ckpt/`` directory (used only during conversion) is not required.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        llm_device: str = "CPU",
        audio_device: str = "CPU",
        asr_device: str = "CPU",
        ov_config: Optional[dict] = None,
        load_asr: bool = False,
    ):
        import torch
        import types
        from omnivoice.models.omnivoice import OmniVoice as _OmniVoice, OmniVoiceConfig
        from omnivoice.utils.duration import RuleDurationEstimator
        from transformers import (
            AutoFeatureExtractor,
            AutoTokenizer,
            HiggsAudioV2TokenizerConfig,
        )

        self.model_dir = Path(model_dir)
        for required in (LLM_FILE, ENCODER_FILE, DECODER_FILE, "config.json", "tokenizer.json", "audio_tokenizer/config.json"):
            if not (self.model_dir / required).exists():
                raise FileNotFoundError(f"Missing {required} in {self.model_dir}. " f"Run convert_omnivoice() first.")

        # Load configs + tokenizers from the OV output dir (NOT from ckpt/).
        cfg = OmniVoiceConfig.from_pretrained(str(self.model_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        audio_cfg = HiggsAudioV2TokenizerConfig.from_pretrained(str(self.model_dir / "audio_tokenizer"))
        feat_extractor = AutoFeatureExtractor.from_pretrained(str(self.model_dir / "audio_tokenizer"))

        # Compile OV sub-models.
        core = ov.Core()
        ov_config = ov_config or {}
        compiled_llm = core.compile_model(self.model_dir / LLM_FILE, llm_device, ov_config)
        compiled_enc = core.compile_model(self.model_dir / ENCODER_FILE, audio_device, ov_config)
        compiled_dec = core.compile_model(self.model_dir / DECODER_FILE, audio_device, ov_config)
        self._ov_llm = _OVTorchLLM(compiled_llm, llm_device)

        # Build the lightweight stand-in. No nn.Module, no random weight init.
        model = _OmniVoiceStandIn()
        model.config = cfg
        model.text_tokenizer = tokenizer
        model.audio_tokenizer = _OVAudioTokenizer(compiled_enc, compiled_dec, audio_cfg)
        model.feature_extractor = feat_extractor
        model.sampling_rate = feat_extractor.sampling_rate
        model.device = torch.device("cpu")
        model.duration_estimator = RuleDurationEstimator()
        model._asr_pipe = None
        # generate() calls self.eval(); make it a no-op.
        model.eval = lambda: None
        # __call__ on the stand-in dispatches the LLM forward to OpenVINO.
        ov_llm = self._ov_llm
        model._ov_call = lambda input_ids, audio_mask, attention_mask: ov_llm(input_ids, audio_mask, attention_mask)

        # Bind the upstream OmniVoice generation methods to the stand-in.
        # This reuses the entire diffusion / chunking / sampling / post-processing
        # pipeline without instantiating an nn.Module.
        for name in (
            "generate",
            "_preprocess_all",
            "_generate_iterative",
            "_generate_chunked",
            "_prepare_inference_inputs",
            "_decode_and_post_process",
            "_post_process_audio",
            "_estimate_target_tokens",
            "_ensure_list",
            "_predict_tokens_with_scoring",
            "create_voice_clone_prompt",
            "transcribe",
        ):
            setattr(model, name, types.MethodType(getattr(_OmniVoice, name), model))

        self._model = model
        self._asr_device = asr_device
        self._asr_pipe = None
        self._ov_whisper_dir = self.model_dir / WHISPER_DIR
        if load_asr:
            self.load_asr_model()

    # --- pass-through API -------------------------------------------------

    @property
    def sampling_rate(self):
        return self._model.sampling_rate

    @property
    def text_tokenizer(self):
        return self._model.text_tokenizer

    @property
    def supported_language_names(self):
        return self._model.supported_language_names

    def supported_language_ids(self):
        return self._model.supported_language_ids()

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, preprocess_prompt=True):
        # The upstream helper calls self.transcribe() if ref_text is None and
        # self._asr_pipe is loaded — keep that behaviour by exposing transcribe.
        # When ref_text is None and ASR isn't loaded yet, lazy-load it.
        if ref_text is None and self._asr_pipe is None:
            self.load_asr_model()
            self._model._asr_pipe = self._asr_pipe  # so OmniVoice.transcribe() works
        return self._model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text, preprocess_prompt=preprocess_prompt)

    def transcribe(self, audio):
        if self._asr_pipe is None:
            self.load_asr_model()
        if isinstance(audio, str):
            return self._asr_pipe(audio)["text"].strip()
        waveform, sr = audio
        import torch

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        waveform = np.squeeze(waveform)
        return self._asr_pipe({"array": waveform, "sampling_rate": sr})["text"].strip()

    def load_asr_model(self):
        """Load the OpenVINO Whisper from ``model_dir/whisper``.

        Voice clone with an explicit ``ref_text`` does NOT need this.
        """
        if self._asr_pipe is not None:
            return
        if not (self._ov_whisper_dir / "openvino_encoder_model.xml").exists():
            raise FileNotFoundError(
                f"OV Whisper not found at {self._ov_whisper_dir}. "
                f"Re-run convert_omnivoice(convert_whisper=True) or pass "
                f"ref_text explicitly to skip auto-transcription."
            )
        print("⌛ Loading OV Whisper ASR on", self._asr_device, "...")
        self._asr_pipe = _OVWhisperASR(self._ov_whisper_dir, self._asr_device)
        # OmniVoice.create_voice_clone_prompt also reads self._asr_pipe directly.
        self._model._asr_pipe = self._asr_pipe
        print("✅ OV Whisper ready on", self._asr_device)

    def generate(self, *args, **kwargs):
        # Lazy-load ASR if the caller passes ref_audio without ref_text -- the
        # upstream _preprocess_all path would otherwise call self.load_asr_model()
        # on the stand-in, which we don't expose there.
        ref_audio = kwargs.get("ref_audio")
        ref_text = kwargs.get("ref_text")
        if ref_audio is not None and ref_text is None and self._asr_pipe is None:
            self.load_asr_model()
            self._model._asr_pipe = self._asr_pipe
        return self._model.generate(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        llm_device: str = "CPU",
        audio_device: Optional[str] = None,
        asr_device: Optional[str] = None,
        ov_config: Optional[dict] = None,
        load_asr: bool = False,
    ) -> "OVOmniVoice":
        """Load an OV-backed OmniVoice from a directory previously produced by
        :func:`convert_omnivoice`."""
        return cls(
            model_dir=model_dir,
            llm_device=llm_device,
            audio_device=audio_device or llm_device,
            asr_device=asr_device or "CPU",
            ov_config=ov_config,
            load_asr=load_asr,
        )
