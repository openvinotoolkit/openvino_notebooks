import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel
from transformers.feature_extraction_utils import BatchFeature

HELPER = Path(__file__).resolve().parents[1] / "notebooks/polyglot-lion/polyglot_helper.py"
spec = importlib.util.spec_from_file_location("polyglot_helper", HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class RecordingProcessor:
    def apply_chat_template(self, messages, **kwargs):
        return str(messages)

    def __call__(self, *, text, audio, **kwargs):
        self.text = text[0] if isinstance(text, list) else text
        self.audio = audio[0] if isinstance(audio, list) else audio
        return BatchFeature({"input_ids": torch.tensor([[42]])})

    def batch_decode(self, *args, **kwargs):
        return [""]


class CaptureModel:
    device = torch.device("cpu")
    dtype = torch.float32

    def generate(self, input_ids, **kwargs):
        return SimpleNamespace(sequences=torch.cat([input_ids, torch.tensor([[151645]])], dim=1))


class ReferenceInputTest(unittest.TestCase):
    def test_audio_and_prompt_match_upstream_transcribe(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.wav"
            for name, rate, stereo in (("normal", 16000, False), ("hot-stereo", 24000, True)):
                with self.subTest(name=name):
                    t = np.arange(rate, dtype=np.float32) / rate
                    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
                    if stereo:
                        audio = np.stack((1.3 * np.sin(2 * np.pi * 440 * t), 0.9 * np.sin(2 * np.pi * 660 * t)), axis=1)
                    sf.write(path, audio, rate, subtype="FLOAT")
                    reference_processor = RecordingProcessor()
                    model = Qwen3ASRModel(backend="transformers", model=CaptureModel(), processor=reference_processor)
                    model.transcribe(audio=str(path), language=None)
                    notebook_processor = RecordingProcessor()
                    _, duration = helper.prepare_inputs(notebook_processor, path)
                    self.assertEqual(notebook_processor.text, reference_processor.text)
                    np.testing.assert_array_equal(notebook_processor.audio, reference_processor.audio)
                    self.assertEqual(duration, 1.0)

    def test_audio_limit_checked_before_decoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.wav"
            sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
            with patch.object(helper, "normalize_audio_input") as normalize:
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    helper.prepare_inputs(RecordingProcessor(), path, max_duration_seconds=0.5)
                normalize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
