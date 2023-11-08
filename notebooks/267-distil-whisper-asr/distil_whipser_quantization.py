import shutil
from contextlib import contextmanager
from datetime import datetime
from itertools import islice
from typing import Any

import numpy as np
import openvino as ov
from pathlib import Path
from datasets import load_dataset
from openvino import Tensor
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from jiwer import wer, wer_standardize

import nncf

core = ov.Core()

device = "CPU"

model_id = "distil-whisper/distil-large-v2"
model_dir = Path(model_id.split("/")[-1])
quantized_model_dir = model_dir / "quantized"

processor = AutoProcessor.from_pretrained(model_id)


COLLECT_CALIBRATION_DATA = False


@contextmanager
def calibration_data_collection():
    global COLLECT_CALIBRATION_DATA
    try:
        COLLECT_CALIBRATION_DATA = True
        yield
    finally:
        COLLECT_CALIBRATION_DATA = False


def convert_to_ov():
    if not model_dir.exists():
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id, export=True, compile=False
        )
        ov_model.half()
        ov_model.save_pretrained(model_dir)
    else:
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_dir, compile=False
        )
    return ov_model


class InferRequestWrapper:
    def __init__(self, request, data_cache):
        self.request = request
        self.data_cache = data_cache

    def __call__(self, *args, **kwargs):
        if COLLECT_CALIBRATION_DATA:
            # self.data_cache.append(args[0])
            self.data_cache.append(*args)
        return self.request(*args, *kwargs)

    def infer(self, inputs: Any = None, shared_memory: bool = False):
        if COLLECT_CALIBRATION_DATA:
            self.data_cache.append(inputs)
        return self.request.infer(inputs, shared_memory)

    def wait(self):
        pass

    def get_tensor(self, name: str):
        return Tensor(self.request.results[name])

    def start_async(
            self,
            inputs: Any = None,
            userdata: Any = None,
            shared_memory: bool = False,
    ):
        if COLLECT_CALIBRATION_DATA:
            self.data_cache.append(inputs)
        self.request.infer(inputs, shared_memory)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.request, attr)


def extract_input_features(sample):
    input_features = processor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    ).input_features
    return input_features


def time_it(obj, fn_name, time_list):
    original_fn = getattr(obj, fn_name)

    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = original_fn(*args, **kwargs)
        end_time = datetime.now()
        time_list.append((end_time - start_time).total_seconds())
        return result

    setattr(obj, fn_name, wrapper)


def collect_calibration_dataset(ov_model, calibration_dataset_size):
    encoder_calibration_data = []
    decoder_calibration_data = []
    ov_model.encoder.request = InferRequestWrapper(ov_model.encoder.request, encoder_calibration_data)
    ov_model.decoder_with_past.request = InferRequestWrapper(ov_model.decoder_with_past.request,
                                                             decoder_calibration_data)

    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    for sample in tqdm(islice(dataset, calibration_dataset_size), desc="Collecting calibration data",
                       total=calibration_dataset_size):
        input_features = extract_input_features(sample)

        with calibration_data_collection():
            ov_model.generate(input_features)

    return encoder_calibration_data, decoder_calibration_data


def quantize(ov_model, calibration_dataset_size, encoder_sq_alpha, decoder_sq_alpha):
    # encoder_calibration_data, decoder_calibration_data = collect_calibration_dataset(ov_model,
    #                                                                                  calibration_dataset_size)
    # print(len(encoder_calibration_data), len(decoder_calibration_data))

    save_dir_name = f"subset{calibration_dataset_size}_enc-sq-{encoder_sq_alpha:.2f}_dec-sq-{decoder_sq_alpha:.2f}_tmp"
    save_dir = quantized_model_dir / save_dir_name
    if not save_dir.exists():
        encoder_calibration_data, decoder_calibration_data = collect_calibration_dataset(ov_model,
                                                                                         calibration_dataset_size)
        print("Quantizing encoder")
        quantized_encoder = nncf.quantize(
            ov_model.encoder.model,
            nncf.Dataset(encoder_calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(encoder_calibration_data),
            fast_bias_correction=True,
            model_type=nncf.ModelType.TRANSFORMER,
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=encoder_sq_alpha)
        )
        ov.save_model(quantized_encoder, save_dir / "openvino_encoder_model.xml")

        print("Quantizing decoder with past")
        quantized_decoder_with_past = nncf.quantize(
            ov_model.decoder_with_past.model,
            nncf.Dataset(decoder_calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(decoder_calibration_data),
            fast_bias_correction=True,
            model_type=nncf.ModelType.TRANSFORMER,
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=decoder_sq_alpha)
        )
        ov.save_model(quantized_decoder_with_past, save_dir / "openvino_decoder_with_past_model.xml")

        shutil.copy(model_dir / "config.json", save_dir / "config.json")
        shutil.copy(model_dir / "openvino_decoder_model.xml", save_dir / "openvino_decoder_model.xml")
        shutil.copy(model_dir / "openvino_decoder_model.bin", save_dir / "openvino_decoder_model.bin")

    quantized_ov_model = OVModelForSpeechSeq2Seq.from_pretrained(save_dir, compile=False)
    quantized_ov_model.to(device)
    quantized_ov_model.compile()
    return quantized_ov_model


def predict(ov_model, n_samples, print_predictions):
    # dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True).take(n_samples)

    whole_infer_times = []
    encoder_infer_times = []
    decoder_infer_times = []
    decoder_with_past_infer_times = []
    time_it(ov_model, "generate", whole_infer_times)
    time_it(ov_model.encoder, "forward", encoder_infer_times)
    time_it(ov_model.decoder, "forward", decoder_infer_times)
    time_it(ov_model.decoder_with_past, "forward", decoder_with_past_infer_times)

    for sample in tqdm(islice(dataset, n_samples), desc="Running", disable=print_predictions,
                       total=n_samples):
        input_features = extract_input_features(sample)
        predicted_ids = ov_model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        if print_predictions:
            print()
            print(f"Reference: {sample['text']}")
            print(f"Result: {transcription[0]}")

    print()
    print(f"Whole inference time. Mean: {np.mean(whole_infer_times):.3f} s. "
          f"Sum: {np.sum(whole_infer_times):.3f} s. Count: {len(whole_infer_times)} calls")
    print(f"Encoder inference time: Mean: {np.mean(encoder_infer_times):.3f} s. "
          f"Sum: {np.sum(encoder_infer_times):.3f} s. Count: {len(encoder_infer_times)} calls")
    print(f"Decoder inference time: Mean: {np.mean(decoder_infer_times):.3f} s. "
          f"Sum: {np.sum(decoder_infer_times):.3f} s. Count: {len(decoder_infer_times)} calls")
    print(f"Decoder with past inference time: "
          f"Mean: {np.mean(decoder_with_past_infer_times):.3f} s. Sum: {np.sum(decoder_with_past_infer_times):.3f} s. "
          f"Count: {len(decoder_with_past_infer_times)} calls")


def validate(ov_model, test_dataset_size=100):
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    dataset = dataset.shuffle(seed=42).take(test_dataset_size)

    ground_truths = []
    predictions = []
    inference_time = []
    for data_item in tqdm(dataset, desc="Measuring performance and accuracy", total=test_dataset_size):
        input_features = extract_input_features(data_item)

        start_time = datetime.now()
        predicted_ids = ov_model.generate(input_features)
        end_time = datetime.now()
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        delta_time = (end_time - start_time).total_seconds()

        print()
        print(data_item["text"])
        print(transcription[0])
        ground_truths.append(data_item["text"])
        predictions.append(transcription[0])
        inference_time.append(delta_time)

    word_accuracy = (1 - wer(ground_truths, predictions, reference_transform=wer_standardize,
                             hypothesis_transform=wer_standardize)) * 100
    mean_inference_time = np.mean(inference_time)
    return mean_inference_time, word_accuracy


ov_model = convert_to_ov()
ov_model.to(device)
ov_model.compile()


quantized_ov_model = quantize(ov_model,
                              calibration_dataset_size=10,
                              encoder_sq_alpha=0.50,
                              decoder_sq_alpha=0.95)

# n_samples = 1
# predict(ov_model, n_samples=n_samples, print_predictions=bool(0))
# predict(quantized_ov_model, n_samples=n_samples, print_predictions=bool(0))

test_size = 50
transcription_time_fp32, accuracy_fp32 = validate(ov_model, test_dataset_size=test_size)
transcription_time_int8, accuracy_int8 = validate(quantized_ov_model, test_dataset_size=test_size)
print(f"Whisper transcription performance speedup: {transcription_time_fp32 / transcription_time_int8:.3f}")
print(f"Whisper transcription word accuracy. FP32: {accuracy_fp32:.2f}%. INT8: {accuracy_int8:.2f}%. "
      f"Accuracy drop :{accuracy_fp32 - accuracy_int8:.2f}%.")
