import datasets
import nncf
import openvino as ov
import time

from contextlib import contextmanager
from jiwer import wer, wer_standardize
from nncf.quantization.range_estimator import (
    RangeEstimatorParameters,
    StatisticsCollectorParameters,
    StatisticsType,
)
from optimum.intel import OVModelForSeq2SeqLM
from optimum.intel.openvino.quantization import InferRequestWrapper
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Dict
from transformers import Pipeline, pipeline, PreTrainedTokenizer

CALIBRATION_DATASET_SIZE = 10


def collect_calibration_data(grammar_corrector_pipe_fp32: Pipeline, calibration_dataset_size: int) -> List[Dict]:
    calibration_data = []
    ov_decoder = grammar_corrector_pipe_fp32.model.decoder_with_past

    # Wrap decoder inference for data collection
    ov_decoder.request = InferRequestWrapper(ov_decoder.request, calibration_data, apply_caching=True)

    # Run inference for data collection
    try:
        calibration_dataset = datasets.load_dataset("jfleg", split="validation")
        calibration_dataset = calibration_dataset.shuffle(seed=42)[:calibration_dataset_size]
        for data_item in tqdm(
            calibration_dataset["sentence"],
            total=calibration_dataset_size,
            desc="Collecting calibration data",
        ):
            grammar_corrector_pipe_fp32(data_item)
    finally:
        ov_decoder.request = ov_decoder.request.request

    return calibration_data


def quantize(
    grammar_corrector_pipe_fp32: Pipeline,
    core: ov.Core,
    quantized_model_path: Path,
    calibration_dataset_size: int,
):
    if quantized_model_path.exists():
        print("Loading quantized model")
        quantized_model = core.read_model(model=quantized_model_path)
    else:
        calibration_data = collect_calibration_data(grammar_corrector_pipe_fp32, calibration_dataset_size)
        ov_decoder = grammar_corrector_pipe_fp32.model.decoder_with_past
        quantized_model = nncf.quantize(
            ov_decoder.model,
            calibration_dataset=nncf.Dataset(calibration_data),
            subset_size=len(calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            advanced_parameters=nncf.AdvancedQuantizationParameters(
                disable_bias_correction=True,
                # Disable bias correction because the model does not contain quantizable operations with bias
                activations_range_estimator_params=RangeEstimatorParameters(
                    # Quantile statistic is employed due to outliers in some activations
                    # This parameter was found useful by quantize_with_accuracy_control method
                    max=StatisticsCollectorParameters(StatisticsType.QUANTILE)
                ),
            ),
        )

        if not quantized_model_path.parent.exists():
            quantized_model_path.parent.mkdir(parents=True)
        ov.save_model(quantized_model, quantized_model_path)

    return quantized_model


def get_quantized_pipeline(
    grammar_corrector_pipe: Pipeline,
    grammar_corrector_tokenizer: PreTrainedTokenizer,
    core: ov.Core,
    grammar_corrector_dir: Path,
    quantized_model_path: Path,
    device: str,
    calibration_dataset_size=CALIBRATION_DATASET_SIZE,
):
    # Get quantized OV model
    quantized_model = quantize(grammar_corrector_pipe, core, quantized_model_path, calibration_dataset_size)

    # Load quantized model into grammar correction pipeline
    grammar_corrector_model_int8 = OVModelForSeq2SeqLM.from_pretrained(grammar_corrector_dir, device=device)
    grammar_corrector_model_int8.decoder_with_past.model = quantized_model
    grammar_corrector_model_int8.decoder_with_past.request = None
    grammar_corrector_model_int8.decoder_with_past._compile()
    grammar_corrector_pipe_int8 = pipeline(
        "text2text-generation",
        model=grammar_corrector_model_int8,
        tokenizer=grammar_corrector_tokenizer,
    )

    return grammar_corrector_pipe_int8


def calculate_compression_rate(model_path_ov, model_path_ov_int8):
    model_size_fp32 = model_path_ov.with_suffix(".bin").stat().st_size / 1024
    model_size_int8 = model_path_ov_int8.with_suffix(".bin").stat().st_size / 1024
    print("Model footprint comparison:")
    print(f"    * FP32 IR model size: {model_size_fp32:.2f} KB")
    print(f"    * INT8 IR model size: {model_size_int8:.2f} KB")
    return model_size_fp32, model_size_int8


def calculate_inference_time_and_accuracy(grammar_corrector_pipe: Pipeline, test_subset_size: int):
    ground_truths = []
    predictions = []
    inference_time = []

    test_dataset = datasets.load_dataset("jfleg", split="test").shuffle(seed=42)[:test_subset_size]
    zipped_dataset = zip(test_dataset["sentence"], test_dataset["corrections"])
    for input_text, references in tqdm(zipped_dataset, total=test_subset_size, desc="Evaluation"):
        # For example, a sample pair may look like:
        # input_text: "For not use car . "
        # references: [ "Not for use with a car . ", "Do not use in the car . ", "Car not for use . "]

        start_time = time.perf_counter()
        corrected_text = grammar_corrector_pipe(input_text)[0]["generated_text"]
        end_time = time.perf_counter()
        delta_time = end_time - start_time

        ground_truths.extend(references)
        predictions.extend([corrected_text] * len(references))
        inference_time.append(delta_time)

    word_accuracy = (
        1
        - wer(
            ground_truths,
            predictions,
            reference_transform=wer_standardize,
            hypothesis_transform=wer_standardize,
        )
    ) * 100
    sum_inference_time = sum(inference_time)
    return sum_inference_time, word_accuracy
