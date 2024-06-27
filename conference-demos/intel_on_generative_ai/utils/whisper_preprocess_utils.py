import io
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from pytube import YouTube
from moviepy.editor import VideoFileClip


def resample_wav(wav_file):
    """
    Resample the wav file to the expected sample rate

    Parameters:
        wav_file: path to input video file
    Returns:
      resampled_audio: mono-channel float audio signal with 16000 Hz sample rate 
                       extracted from wav_file  
    """
    sample_rate, audio = wavfile.read(
        io.BytesIO(open(wav_file, 'rb').read()))
    audio = audio_to_float(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    resampled_audio = resample(audio, sample_rate, 16000)
    return resampled_audio

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
        #segment_lines.append(str(segment["id"] + 1) + "\n")
        time_start = format_timestamp(segment["start"])
        time_end = format_timestamp(segment["end"])
        time_str = f"{time_start} --> {time_end}\n"
        #segment_lines.append(time_str)
        segment_lines.append(segment["text"]) #+ "\n\n"
    return segment_lines