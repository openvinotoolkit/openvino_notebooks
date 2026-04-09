"""
Gradio helper for VoxCPM2 TTS with OpenVINO.
Based on the official VoxCPM2 demo: https://huggingface.co/spaces/openbmb/VoxCPM-Demo
"""

import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import gradio as gr
from scipy.io.wavfile import write as wav_write


def _normalize_audio(wav, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav, dtype=np.float32)
    m = np.max(np.abs(x)) if x.size else 0.0
    if m > 1.0 + 1e-6:
        x = x / (m + 1e-12)
    if clip:
        x = np.clip(x, -1.0, 1.0)
    if x.ndim > 1:
        x = np.mean(x, axis=-1).astype(np.float32)
    return x


def _audio_to_path(audio) -> Optional[str]:
    """Convert Gradio audio input to a temp wav file path."""
    if audio is None:
        return None
    if isinstance(audio, str):
        return audio
    if isinstance(audio, tuple) and len(audio) == 2:
        sr, wav = audio
        wav = _normalize_audio(wav)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_write(f.name, int(sr), wav)
            return f.name
    return None


def make_demo(ov_model):
    """Create Gradio Blocks demo for VoxCPM2 with OpenVINO.

    Args:
        ov_model: OVVoxCPM2Model instance
    Returns:
        Gradio Blocks demo
    """

    def generate_tts(
        text: str,
        control_instruction: str,
        ref_audio,
        use_ultimate_clone: bool,
        prompt_text: str,
        cfg_value: float,
        dit_steps: int,
    ):
        if not text or not text.strip():
            return None, "Error: Please enter text to synthesize."

        text = text.strip()
        control = (control_instruction or "").strip()
        if control and not use_ultimate_clone:
            text = f"({control}){text}"

        ref_path = _audio_to_path(ref_audio)
        prompt_text_clean = prompt_text.strip() if use_ultimate_clone and prompt_text else None

        try:
            start_time = time.time()

            kw = dict(
                text=text,
                cfg_value=float(cfg_value),
                inference_timesteps=int(dit_steps),
            )

            if ref_path and prompt_text_clean:
                kw["prompt_wav_path"] = ref_path
                kw["prompt_text"] = prompt_text_clean
            elif ref_path:
                kw["reference_wav_path"] = ref_path

            wav, sr = ov_model.generate(**kw)

            elapsed = time.time() - start_time
            duration = len(wav) / sr
            rtf = elapsed / max(duration, 0.1)
            status = f"✓ Generated {duration:.1f}s audio in {elapsed:.1f}s " f"(RTF: {rtf:.3f})"
            return (sr, wav), status
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

    def on_toggle_ultimate(checked):
        if checked:
            return gr.update(visible=True, value=""), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True, interactive=True)

    with gr.Blocks(
        title="VoxCPM2 TTS — OpenVINO",
    ) as demo:
        gr.Markdown(
            "# VoxCPM2 Text-to-Speech with OpenVINO™\n\n"
            "Generate speech using three modes:\n"
            "- **Voice Design** — describe a voice style in the Control Instruction\n"
            "- **Controllable Cloning** — upload reference audio + optional style control\n"
            "- **Ultimate Cloning** — upload reference audio + provide its transcript"
        )

        with gr.Row():
            with gr.Column():
                ref_audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="🎤 Reference Audio (optional — upload for cloning)",
                )
                use_ultimate = gr.Checkbox(
                    value=False,
                    label="🎙️ Ultimate Cloning Mode",
                    info="Provide reference transcript for faithful cloning. Disables Control Instruction.",
                )
                prompt_text = gr.Textbox(
                    value="",
                    label="Transcript of Reference Audio",
                    placeholder="Enter the transcript of your reference audio …",
                    lines=2,
                    visible=False,
                )
                control_instruction = gr.Textbox(
                    value="",
                    label="🎛️ Control Instruction (optional)",
                    placeholder="e.g. 年轻女性，温柔甜美 / A warm young woman",
                    lines=2,
                )
                target_text = gr.Textbox(
                    value="VoxCPM2 is a creative multilingual TTS model designed to generate highly realistic speech.",
                    label="✍️ Target Text — the content to speak",
                    lines=3,
                )

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    cfg_slider = gr.Slider(1.0, 3.0, value=2.0, step=0.1, label="CFG guidance scale")
                    dit_steps_slider = gr.Slider(1, 50, value=10, step=1, label="DiT flow-matching steps")

                run_btn = gr.Button("🔊 Generate Speech", variant="primary", size="lg")

            with gr.Column():
                audio_out = gr.Audio(label="Generated Audio")
                status_text = gr.Textbox(label="Status", interactive=False)

        use_ultimate.change(
            fn=on_toggle_ultimate,
            inputs=[use_ultimate],
            outputs=[prompt_text, control_instruction],
        )

        run_btn.click(
            fn=generate_tts,
            inputs=[target_text, control_instruction, ref_audio, use_ultimate, prompt_text, cfg_slider, dit_steps_slider],
            outputs=[audio_out, status_text],
        )

    return demo
