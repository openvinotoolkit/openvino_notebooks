"""
Gradio helper for Qwen3-ASR with OpenVINO.
Based on the official Qwen3-ASR demo: https://huggingface.co/spaces/Qwen/Qwen3-ASR
"""

import base64
import io
import os
import time
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import gradio as gr
import numpy as np
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
    Build HTML with per-token audio slices, using base64 data URLs.
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


def make_demo(ov_model, example_dir=None):
    """
    Create Gradio demo for Qwen3-ASR with OpenVINO.

    Args:
        ov_model: OVQwen3ASRModel instance
        example_dir: Directory containing example audio files (optional)

    Returns:
        Gradio Blocks demo
    """
    lang_choices_disp, lang_map = _build_choices_and_map(SUPPORTED_LANGUAGES)
    lang_choices = ["Auto"] + lang_choices_disp

    def transcribe(audio_upload: Any, lang_disp: str, progress=gr.Progress(track_tqdm=True)):
        """
        Main transcription function.
        """
        if audio_upload is None:
            return "", "", None, ""

        try:
            audio_obj = _parse_audio_any(audio_upload)
        except ValueError as e:
            return "", "", None, f"<div style='color:red'>Error: {str(e)}</div>"

        language = None
        if lang_disp and lang_disp != "Auto":
            language = lang_map.get(lang_disp, lang_disp)

        # Measure inference time
        start_time = time.time()

        # Perform transcription
        results = ov_model.transcribe(
            audio=audio_obj,
            language=language,
            return_time_stamps=False,  # Not supported in OV version
        )

        inference_time = time.time() - start_time

        if not isinstance(results, list) or len(results) != 1:
            return "", "", None, "<div style='color:red'>Unexpected result format.</div>"

        r = results[0]

        # Calculate audio duration
        if isinstance(audio_obj, tuple):
            wav, sr = audio_obj
            audio_duration = len(wav) / sr
        else:
            audio_duration = 0

        metrics = f"Inference time: {inference_time:.2f}s | Audio duration: {audio_duration:.2f}s | RTF: {inference_time/max(audio_duration, 0.1):.3f}"

        return (
            getattr(r, "language", "") or "",
            getattr(r, "text", "") or "",
            None,  # No timestamps in OV version
            metrics,
        )

    # Build Gradio interface
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = """
    .gradio-container {max-width: none !important;}
    .main-title {text-align: center; margin-bottom: 20px;}
    """

    with gr.Blocks(theme=theme, css=css, title="Qwen3-ASR with OpenVINO") as demo:
        gr.Markdown(
            """
# Qwen3-ASR with OpenVINO

**Accelerated by OpenVINO™ Runtime**

Qwen3-ASR is a state-of-the-art automatic speech recognition model that supports **52+ languages and dialects** with high accuracy.
This demo uses OpenVINO for accelerated inference on CPU, GPU, or NPU.

**Features:**
- Multi-language ASR (Chinese, English, Japanese, Korean, and 52+ more languages)
- Hardware acceleration via OpenVINO
- Optimized for Intel hardware
"""
        )

        with gr.Row():
            with gr.Column(scale=2):
                audio_in = gr.Audio(
                    label="Upload Audio",
                    type="numpy",
                    sources=["upload", "microphone"],
                )

                # Add example audio selector
                if example_dir is not None:
                    example_path = Path(example_dir)
                    if example_path.exists() and example_path.is_dir():
                        # Get all audio files from example_dir
                        audio_files = sorted(
                            [str(f) for f in example_path.glob("*.wav")]
                            + [str(f) for f in example_path.glob("*.mp3")]
                            + [str(f) for f in example_path.glob("*.flac")]
                        )
                        if audio_files:
                            gr.Examples(
                                examples=[[f] for f in audio_files],
                                inputs=[audio_in],
                                label="Try Example Audio",
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

        with gr.Row():
            out_ts = gr.JSON(label="Timestamps (JSON)", visible=False)

        # Event handlers
        btn.click(
            transcribe,
            inputs=[audio_in, lang_in],
            outputs=[out_lang, out_text, out_ts, out_metrics],
        )

        gr.Markdown(
            """
---
**Links:** [Qwen3-ASR on Hugging Face](https://huggingface.co/collections/Qwen/qwen3-asr) | [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
"""
        )

    return demo
