import gradio as gr
import time
import sys
import os
import io
import tempfile
import subprocess
import requests
from urllib.parse import urlparse
from pydub import AudioSegment
import logging
import torch
import importlib
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class LogCapture(io.StringIO):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def write(self, s):
        super().write(s)
        self.callback(s)


def trim_audio(audio_path, start_time, end_time):
    """
    Trims an audio file to the specified start and end times.

    Args:
        audio_path (str): Path to the audio file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        str: Path to the trimmed audio file.

        Raises:
            gr.Error: If invalid start or end times are provided.
    """
    try:
        logging.info(f"Trimming audio from {start_time} to {end_time}")
        audio = AudioSegment.from_file(audio_path)
        audio_duration = len(audio) / 1000  # Duration in seconds

        # Default start and end times if None
        start_time = max(0, start_time) if start_time is not None else 0
        end_time = min(audio_duration, end_time) if end_time is not None else audio_duration

        # Validate times
        if start_time >= end_time:
            raise gr.Error("End time must be greater than start time.")

        trimmed_audio = audio[int(start_time * 1000) : int(end_time * 1000)]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            trimmed_audio.export(temp_audio_file.name, format="wav")
            logging.info(f"Trimmed audio saved to: {temp_audio_file.name}")
        return temp_audio_file.name
    except Exception as e:
        logging.error(f"Error trimming audio: {str(e)}")
        raise gr.Error(f"Error trimming audio: {str(e)}")


def save_transcription(transcription):
    """
    Saves the transcription text to a temporary file.

    Args:
        transcription (str): The transcription text.

    Returns:
        str: The path to the transcription file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as temp_file:
        temp_file.write(transcription)
        logging.info(f"Transcription saved to: {temp_file.name}")
        return temp_file.name


def make_demo(ov_model, model_dir):
    def transcribe_audio(audio_input, start_time=None, end_time=None, verbose=False):
        """
        Transcribes audio from a given source using SenseVoice.
        Args:
            audio_input (str): Path to uploaded audio file or recorded audio.
            audio_url (str): URL of audio.
            proxy_url (str): Proxy URL if needed.
            proxy_username (str): Proxy username.
            proxy_password (str): Proxy password.
            pipeline_type (str): Type of pipeline to use ('sensevoice').
            model_id (str): The ID of the model to use.
            download_method (str): Method to use for downloading audio.
            start_time (float, optional): Start time in seconds for trimming audio.
            end_time (float, optional): End time in seconds for trimming audio.
            verbose (bool, optional): Whether to output verbose logging.
        Yields:
            Tuple[str, str, str or None]: Metrics and messages, transcription text, path to transcription file.
        """
        # Initialize variables before try block to ensure they're available in finally
        audio_path = None
        is_temp_file = False
        verbose_messages = ""

        try:
            if verbose:
                logging.getLogger().setLevel(logging.INFO)
            else:
                logging.getLogger().setLevel(logging.WARNING)

            verbose_messages = f"Starting transcription with parameters:\n"

            if verbose:
                yield verbose_messages, "", None
            # Determine the audio source

            if audio_input is not None and len(audio_input) > 0:
                # audio_input is a filepath to uploaded or recorded audio
                audio_path = audio_input
                is_temp_file = False
            else:
                error_msg = "No audio source provided. Please upload an audio file, record audio, or enter a URL."
                logging.error(error_msg)
                yield verbose_messages + error_msg, "", None
                return

            # Convert start_time and end_time to float or None
            start_time = float(start_time) if start_time else None
            end_time = float(end_time) if end_time else None

            if start_time is not None or end_time is not None:
                audio_path = trim_audio(audio_path, start_time, end_time)
                is_temp_file = True  # The trimmed audio is a temporary file
                verbose_messages += f"Audio trimmed from {start_time} to {end_time}\n"

            # Perform the transcription
            start_time_perf = time.time()

            system_prompt = "You are a helpful assistant."
            user_prompt = f"语音转写：<|startofspeech|>!{audio_path}<|endofspeech|>"
            contents_i = []
            contents_i.append({"role": "system", "content": system_prompt})
            contents_i.append({"role": "user", "content": user_prompt})
            contents_i.append({"role": "assistant", "content": "null"})
            print(audio_path)
            res, meta_data = ov_model.inference(
                data_in=[audio_path],
            )
            transcription = rich_transcription_postprocess(res[0]["text"])
            end_time_perf = time.time()

            # Calculate metrics
            transcription_time = end_time_perf - start_time_perf
            audio_file_size = os.path.getsize(audio_path) / (1024 * 1024)

            metrics_output = f"Transcription time: {transcription_time:.2f} seconds\n" f"Audio file size: {audio_file_size:.2f} MB\n"

            # Save the transcription to a file
            transcription_file = save_transcription(transcription)

            # Always yield the final result, regardless of verbose setting
            final_metrics = verbose_messages + metrics_output
            yield final_metrics, transcription, transcription_file

        except Exception as e:
            error_msg = f"An error occurred during transcription: {str(e)}"
            logging.error(error_msg)
            yield verbose_messages + error_msg, "", None

        finally:
            # Clean up temporary audio files
            if audio_path and is_temp_file and os.path.exists(audio_path):
                os.remove(audio_path)

    with gr.Blocks() as iface:
        gr.Markdown("# Audio Transcription")
        gr.Markdown("Transcribe audio using SenseVoice model with multilingual support.")

        with gr.Row():
            audio_input = gr.Audio(label="Upload or Record Audio", sources=["upload", "microphone"], type="filepath")

        transcribe_button = gr.Button("Transcribe")

        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                start_time = gr.Number(label="Start Time (seconds)", value=None, minimum=0)
                end_time = gr.Number(label="End Time (seconds)", value=None, minimum=0)
                verbose = gr.Checkbox(label="Verbose Output", value=False)

        with gr.Row():
            metrics_output = gr.Textbox(label="Transcription Metrics and Verbose Messages", lines=10)
            transcription_output = gr.Textbox(label="Transcription", lines=10)
            transcription_file = gr.File(label="Download Transcription")

        def transcribe_with_progress(*args):
            # The audio_input is now the first argument
            for result in transcribe_audio(*args):
                yield result

        transcribe_button.click(
            transcribe_with_progress, inputs=[audio_input, start_time, end_time, verbose], outputs=[metrics_output, transcription_output, transcription_file]
        )

        examples = gr.Examples(
            examples=[
                [str(model_dir / "example/zh.mp3")],
                [str(model_dir / "example/en.mp3")],
            ],
            inputs=[audio_input],
        )

    return iface
