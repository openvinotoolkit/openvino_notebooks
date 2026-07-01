"""
Gradio helper for Qwen3-ASR with OpenVINO (optimum-intel).
Based on the official Qwen3-ASR demo: https://huggingface.co/spaces/Qwen/Qwen3-ASR
"""

import base64
import io
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import torch
from scipy.io.wavfile import write as wav_write

# Supported languages (same as official Qwen3-ASR)
SUPPORTED_LANGUAGES = [
    "Chinese",
    "Cantonese",
    "English",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
]


def _title_case_display(s: str) -> str:
    """Convert language name to title case display."""
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    """Build dropdown choices and mapping."""
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 mono."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    """
    Accept gradio audio formats and convert to (wav_float32_mono, sr).

    Supports:
        - {"sampling_rate": int, "data": np.ndarray}
        - (sr, np.ndarray) or (np.ndarray, sr)
    """
    if audio is None:
        return None

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    if isinstance(audio, tuple) and len(audio) == 2:
        a0, a1 = audio
        if isinstance(a0, int):
            sr = int(a0)
            wav = _normalize_audio(a1)
            return wav, sr
        if isinstance(a1, int):
            wav = _normalize_audio(a0)
            sr = int(a1)
            return wav, sr

    return None


def _parse_audio_any(audio: Any) -> Union[str, Tuple[np.ndarray, int]]:
    """Parse audio input to either file path or (wav, sr) tuple."""
    if audio is None:
        raise ValueError("Audio is required.")
    at = _audio_to_tuple(audio)
    if at is not None:
        return at
    raise ValueError("Unsupported audio input format.")


def _make_timestamp_html(audio_upload: Any, timestamps: Any) -> str:
    """
    Build HTML with per-word audio slices, using base64 data URLs.
    """
    at = _audio_to_tuple(audio_upload)
    if at is None:
        return "<div style='color:#666'>No audio available for visualization.</div>"
    audio, sr = at

    if not timestamps:
        return "<div style='color:#666'>No timestamps to visualize.</div>"
    if not isinstance(timestamps, list):
        return "<div style='color:#666'>Invalid timestamp format.</div>"

    html_content = """
    <style>
        .word-alignment-container { display: flex; flex-wrap: wrap; gap: 10px; }
        .word-box {
            border: 1px solid #ddd; border-radius: 8px; padding: 10px;
            background-color: #f9f9f9; box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            text-align: center;
        }
        .word-text { font-size: 18px; font-weight: 700; margin-bottom: 5px; }
        .word-time { font-size: 12px; color: #666; margin-bottom: 8px; }
        .word-audio audio { width: 140px; height: 30px; }
        details { border: 1px solid #ddd; border-radius: 6px; padding: 10px; background-color: #f7f7f7; }
        summary { font-weight: 700; cursor: pointer; }
    </style>
    """

    html_content += """
    <details open>
        <summary>Timestamps Visualization (click each word to hear the audio segment)</summary>
        <div class="word-alignment-container" style="margin-top: 14px;">
    """

    for item in timestamps:
        if not isinstance(item, dict):
            continue
        word = str(item.get("text", "") or "")
        start = item.get("start_time", None)
        end = item.get("end_time", None)
        if start is None or end is None:
            continue

        start = float(start)
        end = float(end)
        if end <= start:
            continue

        start_sample = max(0, int(start * sr))
        end_sample = min(len(audio), int(end * sr))
        if end_sample <= start_sample:
            continue

        seg = audio[start_sample:end_sample]
        seg_i16 = (np.clip(seg, -1.0, 1.0) * 32767.0).astype(np.int16)

        mem = io.BytesIO()
        wav_write(mem, sr, seg_i16)
        mem.seek(0)
        b64 = base64.b64encode(mem.read()).decode("utf-8")
        audio_src = f"data:audio/wav;base64,{b64}"

        html_content += f"""
        <div class="word-box">
            <div class="word-text">{word}</div>
            <div class="word-time">{start:.3f}s - {end:.3f}s</div>
            <div class="word-audio">
                <audio controls preload="none" src="{audio_src}"></audio>
            </div>
        </div>
        """

    html_content += "</div></details>"
    return html_content


def save_transcription(transcription: str) -> str:
    """Save transcription text to a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
        f.write(transcription)
        return f.name


def _resample(wav: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample mono float32 audio to target_sr if needed."""
    if sr == target_sr:
        return wav
    import librosa

    return librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)


def make_demo(asr_model, processor, aligner_model=None):
    """
    Create a Gradio demo for Qwen3-ASR with OpenVINO (optimum-intel).

    Args:
        asr_model: an ``OVModelForSpeechSeq2Seq`` loaded from a Qwen3-ASR checkpoint.
        processor: the ``AutoProcessor`` for the model (handles ``apply_transcription_request``,
            ``decode`` and, for the aligner, ``prepare_forced_aligner_inputs`` / ``decode_forced_alignment``).
        aligner_model: optional ``OVModelForQwen3ASRForcedAligner`` for word-level timestamps.
    """
    lang_choices_disp, lang_map = _build_choices_and_map(SUPPORTED_LANGUAGES)
    lang_choices = ["Auto"] + lang_choices_disp

    def transcribe(audio_upload: Any, lang_disp: str, progress=gr.Progress(track_tqdm=True)):
        """Transcribe (and optionally align) the uploaded audio, mirroring the original model API."""
        if audio_upload is None:
            return "", "", gr.update(value=None, visible=False), ""

        try:
            wav, sr = _parse_audio_any(audio_upload)
        except ValueError as e:
            return "", "", gr.update(value=None, visible=False), f"<div style='color:red'>Error: {str(e)}</div>"

        wav = _resample(wav, sr, 16000)
        sr = 16000

        language = None
        if lang_disp and lang_disp != "Auto":
            language = lang_map.get(lang_disp, lang_disp)

        start_time = time.time()

        # 1. Transcribe -- identical flow to the original transformers model.
        inputs = processor.apply_transcription_request(audio=wav, sampling_rate=sr, language=language)
        inputs = inputs.to(asr_model.device)
        generated_ids = asr_model.generate(**inputs, max_new_tokens=256)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        parsed = processor.decode(generated_ids, return_format="parsed")[0]
        detected_language = parsed.get("language", "") or (language or "")
        transcription = parsed.get("transcription", "") or ""

        inference_time = time.time() - start_time

        # 2. Optionally compute word-level timestamps with the forced aligner.
        timestamps = None
        if aligner_model is not None and transcription.strip():
            try:
                aligner_inputs, word_lists = processor.prepare_forced_aligner_inputs(
                    audio=wav,
                    transcript=transcription,
                    language=detected_language or "English",
                )
                aligner_inputs = aligner_inputs.to(aligner_model.device)
                with torch.inference_mode():
                    outputs = aligner_model(**aligner_inputs)
                timestamps = processor.decode_forced_alignment(
                    logits=outputs.logits,
                    input_ids=aligner_inputs["input_ids"],
                    word_lists=word_lists,
                    timestamp_token_id=aligner_model.config.timestamp_token_id,
                )[0]
            except Exception as e:
                timestamps = None
                print(f"Forced alignment failed: {e}")

        audio_duration = len(wav) / sr
        metrics = (
            f"Inference time: {inference_time:.2f}s | " f"Audio duration: {audio_duration:.2f}s | " f"RTF: {inference_time / max(audio_duration, 0.1):.3f}"
        )

        ts_update = gr.update(value=timestamps, visible=True) if timestamps else gr.update(value=None, visible=False)

        return detected_language, transcription, ts_update, metrics

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = """
    .gradio-container {max-width: none !important;}
    .main-title {text-align: center; margin-bottom: 20px;}
    """

    timestamps_note = "- Word-level timestamps via Qwen3-ForcedAligner\n" if aligner_model is not None else ""

    with gr.Blocks(theme=theme, css=css, title="Qwen3-ASR with OpenVINO") as demo:
        gr.Markdown(f"""
# Qwen3-ASR with OpenVINO

**Accelerated by OpenVINO™ Runtime via Optimum Intel**

Qwen3-ASR is a state-of-the-art automatic speech recognition model that supports **52+ languages and dialects** with high accuracy.
This demo uses OpenVINO for accelerated inference on CPU, GPU, or NPU.

**Features:**
- Multi-language ASR (Chinese, English, Japanese, Korean, and 52+ more languages)
{timestamps_note}- Hardware acceleration via OpenVINO
""")

        with gr.Row():
            with gr.Column(scale=2):
                audio_in = gr.Audio(
                    label="Upload Audio",
                    type="numpy",
                    sources=["upload", "microphone"],
                )

                lang_in = gr.Dropdown(
                    label="Language (leave 'Auto' for automatic detection)",
                    choices=lang_choices,
                    value="Auto",
                    interactive=True,
                )
                btn = gr.Button("Transcribe", variant="primary", size="lg")

            with gr.Column(scale=3):
                out_lang = gr.Textbox(label="Detected Language", lines=1, interactive=False)
                out_text = gr.Textbox(label="Transcription Result", lines=10, interactive=False)
                out_metrics = gr.Textbox(label="Inference Metrics", lines=1, interactive=False)

        out_ts_html = gr.HTML(label="Word Timestamps", visible=False)

        def _on_transcribe(audio_upload, lang_disp):
            detected_language, transcription, ts_update, metrics = transcribe(audio_upload, lang_disp)
            # Render timestamp visualization (audio slices per word) when available.
            if ts_update.get("visible") and ts_update.get("value"):
                html = _make_timestamp_html(audio_upload, ts_update["value"])
                ts_html_update = gr.update(value=html, visible=True)
            else:
                ts_html_update = gr.update(value="", visible=False)
            return detected_language, transcription, metrics, ts_html_update

        btn.click(
            _on_transcribe,
            inputs=[audio_in, lang_in],
            outputs=[out_lang, out_text, out_metrics, out_ts_html],
        )

        gr.Markdown("""
---
**Links:** [Qwen3-ASR on Hugging Face](https://huggingface.co/collections/Qwen/qwen3-asr) | [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
""")

    return demo
