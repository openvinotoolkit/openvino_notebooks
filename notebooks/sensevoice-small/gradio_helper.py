import gradio as gr
import librosa
import numpy as np
import numpy.typing as npt
import openvino_genai as ov_genai
import soundfile as sf


def prepare_audio(waveform: npt.ArrayLike, sampling_rate: int, max_seconds: float = 30.0) -> npt.NDArray[np.float32]:
    """Convert a short floating-point waveform to mono 16 kHz audio."""
    if sampling_rate <= 0 or not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("The sampling rate and maximum audio duration must be positive.")
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim not in (1, 2) or audio.size == 0:
        raise ValueError("Provide a nonempty mono or stereo recording.")
    if len(audio) > sampling_rate * max_seconds:
        raise ValueError(f"Use a recording no longer than {max_seconds:g} seconds.")
    if not np.isfinite(audio).all():
        raise ValueError("The recording contains non-finite audio samples.")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sampling_rate != 16000:
        audio = librosa.resample(y=audio, orig_sr=sampling_rate, target_sr=16000)
    return np.ascontiguousarray(audio, dtype=np.float32)


def make_demo(pipe: ov_genai.ASRPipeline) -> gr.Interface:
    def transcribe(audio_path: str | None, language: str, use_itn: bool) -> tuple[str, str, str, str]:
        if audio_path is None:
            raise gr.Error("Upload an audio file or record from the microphone first.")
        try:
            with sf.SoundFile(audio_path) as source:
                audio = source.read(frames=30 * source.samplerate + 1, dtype="float32")
                waveform = prepare_audio(waveform=audio, sampling_rate=source.samplerate)
        except (ValueError, sf.LibsndfileError) as error:
            raise gr.Error(str(error)) from error

        result = pipe.generate(waveform, language=language, use_itn=use_itn)
        metadata = result.metadata[0] if result.metadata else {}
        return result.texts[0], result.languages[0], str(metadata.get("emotion", "Not reported")), str(metadata.get("event", "Not reported"))

    return gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio (up to 30 seconds)"),
            gr.Dropdown(
                choices=[("Automatic", "auto"), ("English", "en"), ("Mandarin", "zh"), ("Cantonese", "yue"), ("Japanese", "ja"), ("Korean", "ko")],
                value="auto",
                label="Language",
            ),
            gr.Checkbox(value=True, label="Inverse text normalization"),
        ],
        outputs=[
            gr.Textbox(label="Transcript"),
            gr.Textbox(label="Language"),
            gr.Textbox(label="Emotion tag"),
            gr.Textbox(label="Audio-event tag"),
        ],
        title="SenseVoiceSmall speech understanding with OpenVINO",
        description="Submit a completed recording for text, language and optional emotion/event predictions. This is not live streaming ASR.",
        flagging_mode="never",
        concurrency_limit=1,
    )
