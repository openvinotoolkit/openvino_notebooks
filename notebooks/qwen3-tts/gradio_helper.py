"""
Gradio helper for Qwen3-TTS with OpenVINO.
Based on the official Qwen3-TTS demo: https://huggingface.co/spaces/Qwen/Qwen3-TTS
"""

import io
import time
import base64
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gradio as gr
from scipy.io.wavfile import write as wav_write


# Supported speakers for CustomVoice model
SPEAKERS = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"]

# Supported languages
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
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


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def save_audio(audio_data: np.ndarray, sample_rate: int) -> str:
    """Save audio to temporary file and return path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_write(f.name, sample_rate, audio_data)
        return f.name


def make_demo(ov_model, model_type: str = "custom_voice"):
    """
    Create Gradio demo for Qwen3-TTS with OpenVINO.

    Args:
        ov_model: OVQwen3TTSModel instance
        model_type: Model type - "custom_voice", "base", or "voice_design"

    Returns:
        Gradio Blocks demo
    """

    def generate_custom_voice(text, language, speaker, instruct, progress=gr.Progress(track_tqdm=True)):
        """Generate speech using CustomVoice model."""
        if not text or not text.strip():
            return None, "Error: Text is required."
        if not speaker:
            return None, "Error: Speaker is required."

        try:
            start_time = time.time()

            wavs, sr = ov_model.generate_custom_voice(
                text=text.strip(),
                language=language if language != "Auto" else None,
                speaker=speaker.lower().replace(" ", "_"),
                instruct=instruct.strip() if instruct else None,
                non_streaming_mode=True,
                max_new_tokens=2048,
            )

            inference_time = time.time() - start_time
            audio_duration = len(wavs[0]) / sr

            status = (
                f"✓ Generation completed!\n"
                f"Inference time: {inference_time:.2f}s | "
                f"Audio duration: {audio_duration:.2f}s | "
                f"RTF: {inference_time/max(audio_duration, 0.1):.3f}"
            )

            return (sr, wavs[0]), status
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

    def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, progress=gr.Progress(track_tqdm=True)):
        """Generate speech using Base (Voice Clone) model."""
        if not target_text or not target_text.strip():
            return None, "Error: Target text is required."

        audio_tuple = _audio_to_tuple(ref_audio)
        if audio_tuple is None:
            return None, "Error: Reference audio is required."

        if not use_xvector_only and (not ref_text or not ref_text.strip()):
            return None, "Error: Reference text is required when 'Use x-vector only' is not enabled."

        try:
            start_time = time.time()

            wavs, sr = ov_model.generate_voice_clone(
                text=target_text.strip(),
                language=language if language != "Auto" else None,
                ref_audio=audio_tuple,
                ref_text=ref_text.strip() if ref_text else None,
                x_vector_only_mode=use_xvector_only,
                max_new_tokens=2048,
            )

            inference_time = time.time() - start_time
            audio_duration = len(wavs[0]) / sr

            status = (
                f"✓ Voice clone completed!\n"
                f"Inference time: {inference_time:.2f}s | "
                f"Audio duration: {audio_duration:.2f}s | "
                f"RTF: {inference_time/max(audio_duration, 0.1):.3f}"
            )

            return (sr, wavs[0]), status
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

    def generate_voice_design(text, language, voice_description, progress=gr.Progress(track_tqdm=True)):
        """Generate speech using Voice Design model."""
        if not text or not text.strip():
            return None, "Error: Text is required."
        if not voice_description or not voice_description.strip():
            return None, "Error: Voice description is required."

        try:
            start_time = time.time()

            wavs, sr = ov_model.generate_voice_design(
                text=text.strip(),
                language=language if language != "Auto" else None,
                instruct=voice_description.strip(),
                non_streaming_mode=True,
                max_new_tokens=2048,
            )

            inference_time = time.time() - start_time
            audio_duration = len(wavs[0]) / sr

            status = (
                f"✓ Voice design completed!\n"
                f"Inference time: {inference_time:.2f}s | "
                f"Audio duration: {audio_duration:.2f}s | "
                f"RTF: {inference_time/max(audio_duration, 0.1):.3f}"
            )

            return (sr, wavs[0]), status
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

    # Build Gradio interface
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = """
    .gradio-container {max-width: none !important;}
    .tab-content {padding: 20px;}
    """

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS with OpenVINO") as demo:
        gr.Markdown(
            """
# Qwen3-TTS with OpenVINO

**Accelerated by OpenVINO™ Runtime**

Qwen3-TTS is a state-of-the-art text-to-speech model that supports multiple languages and speakers.
This demo uses OpenVINO for accelerated inference on CPU, GPU, or NPU.

**Features:**
- Multi-language TTS (Chinese, English, Japanese, Korean, and more)
- Voice cloning from reference audio
- Voice design with natural language descriptions
- Hardware acceleration via OpenVINO
"""
        )

        # Build UI based on model type
        if model_type == "custom_voice":
            # CustomVoice tab
            gr.Markdown("### Text-to-Speech with Predefined Speakers")
            with gr.Row():
                with gr.Column(scale=2):
                    tts_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter the text you want to convert to speech...",
                        value="Hello! Welcome to Text-to-Speech system. This is a demo of our TTS capabilities.",
                    )
                    with gr.Row():
                        tts_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="English",
                            interactive=True,
                        )
                        tts_speaker = gr.Dropdown(
                            label="Speaker",
                            choices=SPEAKERS,
                            value="Ryan",
                            interactive=True,
                        )
                    tts_instruct = gr.Textbox(
                        label="Style Instruction (Optional)",
                        lines=2,
                        placeholder="e.g., Speak in a cheerful and energetic tone",
                    )
                    tts_btn = gr.Button("Generate Speech", variant="primary")

                with gr.Column(scale=2):
                    tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    tts_status = gr.Textbox(label="Status", lines=2, interactive=False)

            tts_btn.click(
                generate_custom_voice,
                inputs=[tts_text, tts_language, tts_speaker, tts_instruct],
                outputs=[tts_audio_out, tts_status],
            )

        elif model_type == "base":
            # Base (Voice Clone) tab
            gr.Markdown("### Clone Voice from Reference Audio")
            with gr.Row():
                with gr.Column(scale=2):
                    clone_ref_audio = gr.Audio(
                        label="Reference Audio (Upload a voice sample to clone)",
                        type="numpy",
                    )
                    clone_ref_text = gr.Textbox(
                        label="Reference Text (Transcript of the reference audio)",
                        lines=2,
                        placeholder="Enter the exact text spoken in the reference audio...",
                    )
                    clone_xvector = gr.Checkbox(
                        label="Use x-vector only (No reference text needed, but lower quality)",
                        value=False,
                    )

                with gr.Column(scale=2):
                    clone_target_text = gr.Textbox(
                        label="Target Text (Text to synthesize with cloned voice)",
                        lines=4,
                        placeholder="Enter the text you want the cloned voice to speak...",
                    )
                    clone_language = gr.Dropdown(
                        label="Language",
                        choices=LANGUAGES,
                        value="Auto",
                        interactive=True,
                    )
                    clone_btn = gr.Button("Clone & Generate", variant="primary")

            with gr.Row():
                clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                clone_status = gr.Textbox(label="Status", lines=2, interactive=False)

            clone_btn.click(
                generate_voice_clone,
                inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector],
                outputs=[clone_audio_out, clone_status],
            )

        elif model_type == "voice_design":
            # Voice Design tab
            gr.Markdown("### Create Custom Voice with Natural Language")
            with gr.Row():
                with gr.Column(scale=2):
                    design_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter the text you want to convert to speech...",
                        value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                    )
                    design_language = gr.Dropdown(
                        label="Language",
                        choices=LANGUAGES,
                        value="Auto",
                        interactive=True,
                    )
                    design_instruct = gr.Textbox(
                        label="Voice Description",
                        lines=3,
                        placeholder="Describe the voice characteristics you want...",
                        value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
                    )
                    design_btn = gr.Button("Generate with Custom Voice", variant="primary")

                with gr.Column(scale=2):
                    design_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    design_status = gr.Textbox(label="Status", lines=2, interactive=False)

            design_btn.click(
                generate_voice_design,
                inputs=[design_text, design_language, design_instruct],
                outputs=[design_audio_out, design_status],
            )

        gr.Markdown(
            """
---
**Links:** [Qwen3-TTS on Hugging Face](https://huggingface.co/collections/Qwen/qwen3-tts) | [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
"""
        )

    return demo
