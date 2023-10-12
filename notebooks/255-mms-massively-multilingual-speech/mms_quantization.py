from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

import nncf
import numpy as np
import torch
import openvino as ov

from datasets import load_dataset
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from transformers import Wav2Vec2ForCTC, AutoProcessor


SAMPLE_LANG = ['german', 'dutch', 'french', 'spanish', 'italian', 'portuguese', 'polish'][0]
LANG_ID = {'german': 'deu', 'french': 'fra'}[SAMPLE_LANG]
MAX_SEQ_LENGTH = 30480

model_lid_id = "facebook/mms-lid-126"
lid_processor = AutoFeatureExtractor.from_pretrained(model_lid_id)
lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_lid_id)

model_asr_id = "facebook/mms-1b-all"
asr_processor = AutoProcessor.from_pretrained(model_asr_id)
asr_model = Wav2Vec2ForCTC.from_pretrained(model_asr_id)
asr_processor.tokenizer.set_target_lang(LANG_ID)
asr_model.load_adapter(LANG_ID)

core = ov.Core()
device = "CPU"

lid_model_xml_path = Path('models/ov_lid_model.xml')
compressed_lid_model_xml_path = Path('models/ov_lid_model_c.xml')
# quantized_lid_model_xml_path = Path('models/ov_lid_model_quantized.xml')
quantized_lid_model_xml_path = Path('models/ov_lid_model_q-p.xml')
asr_model_xml_path = Path(f'models/ov_asr_{LANG_ID}_model.xml')
compressed_asr_model_xml_path = Path(f'models/ov_asr_{LANG_ID}_model_c.xml')
# quantized_asr_model_xml_path = Path(f'models/ov_asr_{LANG_ID}_model_quantized.xml')
quantized_asr_model_xml_path = Path(f'models/ov_asr_{LANG_ID}_model_q-p.xml')

mls = load_dataset("facebook/multilingual_librispeech", SAMPLE_LANG, split="test", streaming=True)


def detect_lang(compiled_lid_model, audio_data):
    inputs = lid_processor(audio_data, sampling_rate=16_000, return_tensors="pt")

    save_input_path = Path("models/inputs/input_lid.npy")
    if not save_input_path.exists():
        np.save(save_input_path, inputs['input_values'].numpy())
    outputs = compiled_lid_model(inputs['input_values'])[0]

    lang_id = torch.argmax(torch.from_numpy(outputs), dim=-1)[0].item()
    detected_lang = lid_model.config.id2label[lang_id]

    return detected_lang


def recognize_audio(compiled_asr_model, src_audio):
    inputs = asr_processor(src_audio, sampling_rate=16_000, return_tensors="pt")

    save_input_path = Path("models/inputs/input_asr.npy")
    if not save_input_path.exists():
        np.save(save_input_path, inputs['input_values'].numpy())
    outputs = compiled_asr_model(inputs['input_values'])[0]

    ids = torch.argmax(torch.from_numpy(outputs), dim=-1)[0]
    transcription = asr_processor.decode(ids)

    return transcription


def get_lid_model(model_path):
    input_values = torch.zeros([1, MAX_SEQ_LENGTH], dtype=torch.float)
    # attention_mask = torch.zeros([1, MAX_SEQ_LENGTH], dtype=torch.int32)

    if not model_path.exists() and model_path == lid_model_xml_path:
        lid_model_xml_path.parent.mkdir(parents=True, exist_ok=True)
        converted_model = ov.convert_model(lid_model, example_input={'input_values': input_values})
        ov.save_model(converted_model, lid_model_xml_path)
    compiled_lid_model = core.compile_model(model_path, device_name=device)
    return compiled_lid_model


def get_asr_model(model_path):
    input_values = torch.zeros([1, MAX_SEQ_LENGTH], dtype=torch.float)
    if not model_path.exists() and model_path == asr_model_xml_path:
        asr_model_xml_path.parent.mkdir(parents=True, exist_ok=True)
        converted_model = ov.convert_model(asr_model, example_input={'input_values': input_values})
        ov.save_model(converted_model, asr_model_xml_path)
    compiled_asr_model = core.compile_model(model_path, device_name=device)
    return compiled_asr_model


# print(core.read_model(lid_model_xml_path).inputs)
# print(core.read_model(asr_model_xml_path).inputs)
# exit(0)

mls = iter(mls)  # make it iterable
example = next(mls)  # get one example

compiled_lid_model = get_lid_model(lid_model_xml_path)
# compiled_lid_model = get_lid_model(compressed_lid_model_xml_path)
# compiled_lid_model = get_lid_model(quantized_lid_model_xml_path)
start_time = datetime.now()
lang_id = detect_lang(compiled_lid_model, example['audio']['array'])
print(f"Language detection: {(datetime.now() - start_time).total_seconds()}")
print(lang_id, LANG_ID)
compiled_asr_model = get_asr_model(asr_model_xml_path)
# compiled_asr_model = get_asr_model(compressed_asr_model_xml_path)
# compiled_asr_model = get_asr_model(quantized_asr_model_xml_path)
start_time = datetime.now()
transcription = recognize_audio(compiled_asr_model, example['audio']['array'])
print(f"Speech recognition: {(datetime.now() - start_time).total_seconds()}")
print(example["text"])
print(transcription)


# compressed_lid_model = nncf.compress_weights(core.read_model(lid_model_xml_path))
# ov.save_model(compressed_lid_model, compressed_lid_model_xml_path)
# compressed_asr_model = nncf.compress_weights(core.read_model(asr_model_xml_path))
# ov.save_model(compressed_asr_model, compressed_asr_model_xml_path)

calibration_data = []
for i in range(1):
    data = asr_processor(next(mls)['audio']['array'], sampling_rate=16_000, return_tensors="pt")["input_values"]
    calibration_data.append(data)

quantized_lid_model = nncf.quantize(
    core.read_model(lid_model_xml_path),
    calibration_dataset=nncf.Dataset(calibration_data),
    # preset=nncf.QuantizationPreset.MIXED,
    preset=nncf.QuantizationPreset.PERFORMANCE,
    subset_size=len(calibration_data),
    fast_bias_correction=True,
    model_type=nncf.ModelType.TRANSFORMER
)
ov.save_model(quantized_lid_model, quantized_lid_model_xml_path)

quantized_asr_model = nncf.quantize(
    core.read_model(asr_model_xml_path),
    calibration_dataset=nncf.Dataset(calibration_data),
    # preset=nncf.QuantizationPreset.MIXED,
    preset=nncf.QuantizationPreset.PERFORMANCE,
    subset_size=len(calibration_data),
    fast_bias_correction=True,
    model_type=nncf.ModelType.TRANSFORMER
)
ov.save_model(quantized_asr_model, quantized_asr_model_xml_path)
