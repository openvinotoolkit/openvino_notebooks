import pathlib

from openvino.runtime import Tensor

from pytube import YouTube
import io
import numpy as np
from scipy.io import wavfile
from moviepy.editor import VideoFileClip


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


def download_video(base_dir: pathlib.Path, link, subdir="", resolution="high"):
    output_file = base_dir / "videos" / subdir / f"{link.split('/')[-1]}.mp4"
    if not output_file.exists():
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)
        print(f"Downloading video {link} started")
        yt = YouTube(link)
        if resolution == "high":
            yt.streams.get_highest_resolution().download(filename=output_file)
        elif resolution == "low":
            yt.streams.get_lowest_resolution().download(filename=output_file)
        else:
            raise Exception("Unknown resolution option")
        print(f"Video saved to {output_file}")
    return output_file


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
    audio_file = pathlib.Path(str(video_file).replace('.mp4', '.wav'))
    if not audio_file.exists():
        input_video = VideoFileClip(str(video_file))
        input_video.audio.write_audiofile(audio_file, verbose=False, logger=None)
    sample_rate, audio = wavfile.read(
        io.BytesIO(open(audio_file, 'rb').read()))
    audio = audio_to_float(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    resampled_audio = resample(audio, sample_rate, 16000)
    return resampled_audio


def convert_input_data_to_np(input_data):
    converted_input_data = []
    for it in input_data:
        if isinstance(it, dict):
            d = {}
            for k, v in it.items():
                d[k] = v.data
        elif isinstance(it, list):
            converted_input_data.append(convert_input_data_to_np(it))
            continue
        else:
            d = it.data
        converted_input_data.append(d)
    return converted_input_data


def convert_input_data_to_ov_tensor(input_data):
    converted_input_data = []
    for it in input_data:
        if isinstance(it, dict):
            d = {}
            for k, v in it.items():
                d[k] = Tensor(v, shared_memory=False)
        else:
            d = it
        converted_input_data.append(d)
    return converted_input_data


def format_timestamp(seconds: float):
    """
    format time in srt-file excpected format
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


def transcribe_trimmed(model, audio, trim=True, **kwargs):
    audio_duration = audio.shape[0] / 16000

    result = model.transcribe(audio, **kwargs)
    if trim:
        new_segments = []
        for segment in result["segments"]:
            # if segment["start"] < audio_duration:
            if segment["end"] <= audio_duration:
                new_segments.append(segment)
        result["segments"] = new_segments
        result["text"] = " ".join(segment["text"] for segment in result["segments"])
    return result
