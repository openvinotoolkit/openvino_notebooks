import argparse
from pathlib import Path

from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor

MODEL_MAPPING = {
    "distil-whisper-large-v2": "distil-whisper/distil-large-v2",
}


def convert_asr_model(model_type: str, precision: str, model_dir: Path) -> Path:
    """
    Convert speech-to-text model

    Params:
        model_type: selected mode type and size
        precision: model precision
        model_dir: dir to export model
    Returns:
       Path to exported model dir
    """

    output_dir = model_dir / model_type
    model_name = MODEL_MAPPING[model_type]

    if precision == "int8":
        output_dir = output_dir.with_name(output_dir.name + "-INT8")

        # use Optimum-Intel to directly quantize weights of the ASR model into INT8
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_name, export=True, compile=False, load_in_8bit=True)
    else:
        output_dir = output_dir.with_name(output_dir.name + "-FP16")

        # load model and convert it to OpenVINO
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_name, export=True, compile=False, load_in_8bit=False)
        ov_model.half()

    ov_model.save_pretrained(output_dir)

    # export also processor
    asr_processor = AutoProcessor.from_pretrained(model_name)
    asr_processor.save_pretrained(output_dir)

    return Path(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_model_type", type=str, choices=["distil-whisper-large-v2"],
                        default="distil-whisper-large-v2", help="Speech recognition model to be converted")
    parser.add_argument("--precision", type=str, default="int8", choices=["fp16", "int8"], help="Model precision")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
    args = parser.parse_args()

    convert_asr_model(args.asr_model_type, args.precision, Path(args.model_dir))
