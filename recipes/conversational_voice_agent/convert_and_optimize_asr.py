import argparse
from pathlib import Path

import openvino as ov
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor


MODEL_NAME = "distil-whisper/distil-large-v2"

def convert_asr_model(use_quantization:bool, model_dir: Path) -> Path:
    """
    Convert speech-to-text model

    Params:
        use_quantization: whether quantize weights to INT8
        model_dir: dir to export model
    Returns:
       Path to exported model dir
    """

    # load model and convert it to OpenVINO
    output_dir = model_dir / (MODEL_NAME.rsplit ("/")[-1] + "-FP16")

    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, export=True, compile=False, load_in_8bit=False)
    ov_model.half()
    ov_model.save_pretrained(output_dir)

    if use_quantization: 
        # Use Optimum-Intel to directly quantize weights of the ASR model into INT8

        quantized_distil_model_path = model_dir / (MODEL_NAME.rsplit ("/")[-1] + "-INT8")
        quantized_ov_model = OVModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, export=True, compile=False, load_in_8bit=True)
        quantized_ov_model.save_pretrained(quantized_distil_model_path)
        output_dir = quantized_distil_model_path

    # export also processor
    asr_processor = AutoProcessor.from_pretrained(MODEL_NAME)
    asr_processor.save_pretrained(output_dir)

    return Path(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_quantization", default=True, help="Choose if to quantize the ASR model")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
    args = parser.parse_args()

    convert_asr_model(args.use_quantization, Path(args.model_dir))
