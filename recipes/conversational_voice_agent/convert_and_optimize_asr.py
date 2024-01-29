import argparse
from pathlib import Path

import openvino as ov
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor
from datasets import load_dataset

from itertools import islice
from typing import List, Any
from openvino import Tensor
import shutil
import nncf
from tqdm.notebook import tqdm
import gc

MODEL_NAME = "distil-whisper/distil-large-v2"

# We use post-traning quantization with NNCF to quantize the mode, which contains the following steps:
    #1: Create a calibration dataset for quantization.
    #2: Run nncf.quantize to obtain quantized encoder and decoder models.
    #3: Serialize the INT8 model using openvino.save_model function.

# Step1: Since we quantize whisper encoder and decoder separately, we need to prepare a calibration dataset for each of the models. 
# We define a InferRequestWrapper class that will intercept model inputs and collect them to a list. 
# Then we run model inference on some small amount of audio samples. Generally, increasing the calibration dataset size improves quantization quality.
class InferRequestWrapper:
    def __init__(self, request, data_cache: List):
        self.request = request
        self.data_cache = data_cache

    def __call__(self, *args, **kwargs):
        self.data_cache.append(*args)
        return self.request(*args, **kwargs)

    def infer(self, inputs: Any = None, shared_memory: bool = False):
        self.data_cache.append(inputs)
        return self.request.infer(inputs, shared_memory)

    def start_async(
        self,
        inputs: Any = None,
        userdata: Any = None,
        share_inputs: bool = False,
    ):
        self.data_cache.append(inputs)
        self.request.infer(inputs, share_inputs)

    def wait(self):
        pass

    def get_tensor(self, name: str):
        return Tensor(self.request.results[name])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.request, attr)


def collect_calibration_dataset(ov_model, calibration_dataset_size):
    # Overwrite model request properties, saving the original ones for restoring later
    original_encoder_request = ov_model.encoder.request
    original_decoder_with_past_request = ov_model.decoder_with_past.request
    encoder_calibration_data = []
    decoder_calibration_data = []
    ov_model.encoder.request = InferRequestWrapper(original_encoder_request, encoder_calibration_data)
    ov_model.decoder_with_past.request = InferRequestWrapper(original_decoder_with_past_request,
                                                             decoder_calibration_data)

    calibration_dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
    for sample in tqdm(islice(calibration_dataset, calibration_dataset_size), desc="Collecting calibration data",
                       total=calibration_dataset_size):
        input_features = extract_input_features(sample)
        ov_model.generate(input_features)

    ov_model.encoder.request = original_encoder_request
    ov_model.decoder_with_past.request = original_decoder_with_past_request

    return encoder_calibration_data, decoder_calibration_data


# Quantize Distil-Whisper encoder and decoder-with-past models. 
def quantize(ov_model, calibration_dataset_size, quantized_distil_model_path, ov_config, output_dir):
    if not quantized_distil_model_path.exists():
        encoder_calibration_data, decoder_calibration_data = collect_calibration_dataset(
            ov_model, calibration_dataset_size
        )
        print("Quantizing encoder")
        quantized_encoder = nncf.quantize(
            ov_model.encoder.model,
            nncf.Dataset(encoder_calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(encoder_calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.50)
        )
        ov.save_model(quantized_encoder, quantized_distil_model_path / "openvino_encoder_model.xml")
        del quantized_encoder
        del encoder_calibration_data
        gc.collect()

        print("Quantizing decoder with past")
        quantized_decoder_with_past = nncf.quantize(
            ov_model.decoder_with_past.model,
            nncf.Dataset(decoder_calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(decoder_calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.95)
        )
        ov.save_model(quantized_decoder_with_past, quantized_distil_model_path / "openvino_decoder_with_past_model.xml")
        del quantized_decoder_with_past
        del decoder_calibration_data
        gc.collect()

        # Copy the config file and the first-step-decoder manually
        shutil.copy(output_dir / "config.json", quantized_distil_model_path / "config.json")
        shutil.copy(output_dir / "openvino_decoder_model.xml", quantized_distil_model_path / "openvino_decoder_model.xml")
        shutil.copy(output_dir / "openvino_decoder_model.bin", quantized_distil_model_path / "openvino_decoder_model.bin")


# Prepare input sample
def extract_input_features(sample):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    input_features = processor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    ).input_features
    return input_features


def convert_asr_model(use_quantization:bool, model_dir: Path) -> Path:
    """
    Convert speech-to-text model

    Params:
        model_dir: dir to export model
    Returns:
       Path to exported model dir
    """

    # load model and convert it to OpenVINO
    output_dir = model_dir / (MODEL_NAME.rsplit ("/")[-1] + "-FP16")
    ov_config = {"CACHE_DIR": ""}

    if not output_dir.exists():
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME, ov_config=ov_config, export=True, compile=False, load_in_8bit=False
        )
        ov_model.half()
        ov_model.save_pretrained(output_dir)
    else:
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            output_dir, ov_config=ov_config, compile=False
        )

    asr_processor = AutoProcessor.from_pretrained(MODEL_NAME)


    if use_quantization: 
        # Use post-training quantization of NNCF to quantize the ASR model

        CALIBRATION_DATASET_SIZE = 50
        quantized_distil_model_path = model_dir / (MODEL_NAME.rsplit ("/")[-1] + "-INT8")
        ov_model.to("AUTO")
        ov_model.compile()
        quantize(ov_model, CALIBRATION_DATASET_SIZE, quantized_distil_model_path, ov_config, output_dir)
        output_dir = quantized_distil_model_path

    # export also processor
    asr_processor.save_pretrained(output_dir)

    return Path(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_quantization", default=True, help="Choose if to quantize the ASR model")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
    args = parser.parse_args()

    convert_asr_model(args.use_quantization, Path(args.model_dir))
