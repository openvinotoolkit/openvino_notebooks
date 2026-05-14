import math
from pathlib import Path
from typing import Optional

import gradio as gr
import requests
import time

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip
from transformers.pipelines.audio_utils import ffmpeg_read

audio_en_example_path = Path("en_example.wav")
audio_ml_example_path = Path("ml_example.wav")

if not audio_en_example_path.exists():
    r = requests.get("https://huggingface.co/spaces/distil-whisper/whisper-vs-distil-whisper/resolve/main/assets/example_1.wav", timeout=30)
    with open(audio_en_example_path, "wb") as f:
        f.write(r.content)

if not audio_ml_example_path.exists():
    r = requests.get("https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jeanNL.wav", timeout=30)
    with open(audio_ml_example_path, "wb") as f:
        f.write(r.content)


MAX_AUDIO_MINS = 30


def get_audio(video_file):
    """Extract audio signal from a given video file."""
    input_video = VideoFileClip(str(video_file))
    duration = input_video.duration
    audio_file = Path(video_file).stem + ".wav"
    input_video.audio.write_audiofile(audio_file, logger=None)
    with open(audio_file, "rb") as f:
        inputs = f.read()
    audio = ffmpeg_read(inputs, 16000)
    return {"raw": audio, "sampling_rate": 16000}, duration


def format_timestamp(seconds: float):
    """Format time in SRT-file expected format."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    return (f"{hours}:" if hours > 0 else "00:") + f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def prepare_srt(transcription, filter_duration=None):
    """Format transcription into SRT file format."""
    segment_lines = []
    for idx, segment in enumerate(transcription):
        start_ts = segment.start_ts
        end_ts = segment.end_ts
        if end_ts == -1:
            end_ts = filter_duration
        if filter_duration is not None and (start_ts >= math.floor(filter_duration) or end_ts > math.ceil(filter_duration) + 1):
            break
        segment_lines.append(str(idx + 1) + "\n")
        time_start = format_timestamp(start_ts)
        time_end = format_timestamp(end_ts)
        segment_lines.append(f"{time_start} --> {time_end}\n")
        segment_lines.append(segment.text + "\n\n")
    return segment_lines


def make_demo(ov_pipe, model_id: str, sample_video: Optional[Path] = None):
    multilingual = not model_id.endswith(".en")

    def transcribe_audio(inputs, task="Transcribe", language=""):
        if not multilingual and task == "Translate":
            raise gr.Error("The model only supports English. The task 'translate' could not be applied.")

        if inputs is None:
            raise gr.Error("No audio file submitted! Please record or upload an audio file before submitting your request.")

        with open(inputs, "rb") as f:
            raw_inputs = f.read()

        audio = ffmpeg_read(raw_inputs, 16000)
        audio_length_mins = len(audio) / 16000 / 60

        if audio_length_mins > MAX_AUDIO_MINS:
            raise gr.Error(
                f"To ensure fair usage of the Space, the maximum audio length permitted is {MAX_AUDIO_MINS} minutes. "
                f"Got an audio of length {round(audio_length_mins, 3)} minutes."
            )

        generate_kwargs = {}
        if task == "Translate":
            generate_kwargs["task"] = "translate"
            if language and language != "auto":
                generate_kwargs["language"] = language

        start_time = time.time()
        ov_text = ov_pipe.generate(audio.copy(), **generate_kwargs)
        ov_time = round(time.time() - start_time, 2)

        return ov_text, ov_time

    def transcribe_video(video_path, task):
        if video_path is None:
            raise gr.Error("No video file submitted! Please upload a video file before submitting your request.")

        data_path = Path(video_path)
        inputs, duration = get_audio(data_path)

        frame_num = len(inputs["raw"]) / 16000
        if frame_num > 30:
            config = ov_pipe.get_generation_config()
            chunk_num = math.ceil(frame_num / 30)
            config.max_length = chunk_num * config.max_length
            ov_pipe.set_generation_config(config)

        transcription = ov_pipe.generate(inputs["raw"], task=task.lower(), return_timestamps=True).chunks
        srt_lines = prepare_srt(transcription, duration)
        srt_path = data_path.with_suffix(".srt")
        with srt_path.open("w") as f:
            f.writelines(srt_lines)

        return [str(data_path), str(srt_path)]

    # Build examples
    audio_examples = [[str(audio_en_example_path), ""]]
    if multilingual:
        audio_examples.append([str(audio_ml_example_path), "<|fr|>"])

    with gr.Blocks() as demo:
        gr.HTML("""
            <div style="text-align: center; max-width: 700px; margin: 0 auto;">
                <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                    OpenVINO Whisper demo
                </h1>
                <p style="font-size: 14px;">Automatic speech recognition and video subtitle generation using OpenVINO Generate API</p>
            </div>
            """)

        with gr.Tabs():
            # === Tab 1: Audio Transcription ===
            with gr.TabItem("Audio Transcription"):
                audio = gr.components.Audio(type="filepath", label="Audio input")
                language = gr.components.Textbox(
                    label="Language",
                    info="List of available languages can be found in generation_config.lang_to_id dictionary. Example: <|en|>. Empty string means autodetection.",
                    value="",
                    visible=multilingual,
                )
                with gr.Row():
                    button_transcribe = gr.Button("Transcribe")
                    button_translate = gr.Button("Translate", visible=multilingual)
                with gr.Row():
                    infer_time = gr.components.Textbox(label="Generation time (s)")
                with gr.Row():
                    result = gr.components.Textbox(label="Transcription result")
                button_transcribe.click(
                    fn=transcribe_audio,
                    inputs=[audio, button_transcribe, language],
                    outputs=[result, infer_time],
                )
                button_translate.click(
                    fn=transcribe_audio,
                    inputs=[audio, button_translate, language],
                    outputs=[result, infer_time],
                )
                gr.Markdown("## Examples")
                gr.Examples(
                    audio_examples,
                    inputs=[audio, language],
                    outputs=[result, infer_time],
                    fn=transcribe_audio,
                    cache_examples=False,
                )

            # === Tab 2: Video Subtitles ===
            with gr.TabItem("Video Subtitles"):
                gr.Markdown(
                    "Upload a video file to generate `.srt` subtitle file. "
                    "If the video is longer than 30 seconds, `max_length` will be increased automatically."
                )
                video_input = gr.Video(label="Video input")
                video_task = gr.Radio(["Transcribe", "Translate"], value="Transcribe", label="Task")
                video_button = gr.Button("Generate subtitles")
                video_output = gr.Video(label="Video with subtitles")
                video_button.click(
                    fn=transcribe_video,
                    inputs=[video_input, video_task],
                    outputs=[video_output],
                )
                if sample_video and sample_video.exists():
                    gr.Examples(
                        [[str(sample_video), "Transcribe"]],
                        inputs=[video_input, video_task],
                        outputs=[video_output],
                        fn=transcribe_video,
                        cache_examples=False,
                    )

    return demo
