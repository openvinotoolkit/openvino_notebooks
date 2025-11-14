import gradio as gr
import torchaudio
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
import librosa
import openvino as ov
import numpy as np


def make_demo(pipe):
    # Load the classifier model
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")

    def f2embed(wav_file, classifier, size_embed):

        signal, fs = stereo_to_mono(wav_file)
        if signal is None:
            return None
        # print(fs, "FS")
        if fs != 16000:
            signal, fs = resample_to_16000(signal, fs)
            if signal is None:
                return None
        assert fs == 16000, fs
        with torch.no_grad():
            embeddings = classifier.encode_batch(signal)
            embeddings = F.normalize(embeddings, dim=2)
            embeddings = embeddings.squeeze().cpu().numpy()
        assert embeddings.shape[0] == size_embed, embeddings.shape[0]
        return embeddings

    def stereo_to_mono(wav_file):
        try:
            signal, fs = torchaudio.load(wav_file)
            signal_np = signal.numpy()
            if signal_np.shape[0] == 2:  # If stereo
                signal_mono = librosa.to_mono(signal_np)
                signal_mono = torch.from_numpy(signal_mono).unsqueeze(0)
            else:
                signal_mono = signal  # Already mono
            print(f"Converted to mono: {signal_mono.shape}")
            return signal_mono, fs
        except Exception as e:
            print(f"Error in stereo_to_mono: {e}")
            return None, None

    def resample_to_16000(signal, original_sr):
        try:
            signal_np = signal.numpy().flatten()
            signal_resampled = librosa.resample(signal_np, orig_sr=original_sr, target_sr=16000)
            signal_resampled = torch.from_numpy(signal_resampled).unsqueeze(0)
            print(f"Resampled to 16000 Hz: {signal_resampled.shape}")
            return signal_resampled, 16000
        except Exception as e:
            print(f"Error in resample_to_16000: {e}")
            return None, None

    def process_audio(wav_file, text):
        try:
            # Extract speaker embeddings
            speaker_embeddings = f2embed(wav_file, classifier, 512)
            if speaker_embeddings is not None:
                embeddings = ov.Tensor(np.array(speaker_embeddings, dtype=np.float32).reshape(1, -1))
                result = pipe.generate(text, embeddings)
            else:
                result = pipe.generate(text)
            speech = result.speeches[0]

            return speech.data[0], 16000
        except Exception as e:
            print(f"Error in process_audio: {e}")
            return None, "Error in audio processing"

    # Gradio interface
    def gradio_interface(text, wav_file):
        try:
            processed_audio, rate = process_audio(wav_file, text)
            if processed_audio is None:
                return "Error occurred during processing"
            return (rate, processed_audio)
        except Exception as e:
            print(f"Error in gradio_interface: {e}")
            return "Error occurred during processing"

    # Create Gradio interface
    gr_interface = gr.Interface(
        fn=gradio_interface,
        inputs=[gr.Textbox(lines=2, placeholder="Enter text here..."), gr.Audio(type="filepath")],
        outputs=gr.Audio(type="numpy"),
        title="OpenVNIO Text-to-Speech with Speaker Embeddings",
        description="Upload a speaker audio file and enter text to convert the text to speech using the speaker's voice.",
        flagging_mode="never",
        examples=[
            ["It is not in the stars to hold our destiny but in ourselves."],
            ["The octopus and Oliver went to the opera in October."],
            ["She sells seashells by the seashore. I saw a kitten eating chicken in the kitchen."],
            ["Brisk brave brigadiers brandished broad bright blades, blunderbusses, and bludgeons—balancing them badly."],
            ["A synonym for cinnamon is a cinnamon synonym."],
            [
                "How much wood would a woodchuck chuck if a woodchuck could chuck wood? He would chuck, he would, as much as he could, and chuck as much wood as a woodchuck would if a woodchuck could chuck wood."
            ],
        ],
    )
    return gr_interface
