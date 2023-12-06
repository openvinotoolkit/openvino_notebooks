import argparse
from pathlib import Path

from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor

MODEL_NAME = "distil-whisper/distil-large-v2"


def convert_asr_model(model_dir: Path) -> Path:
    """
    Convert speech-to-text model

    Params:
        model_dir: dir to export model
    Returns:
       Path to exported model dir
    """
    output_dir = model_dir / (MODEL_NAME.rsplit ("/")[-1] + "-FP16")

    # load model and convert it to OpenVINO
    model = OVModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, export=True, compile=False)
    # change precision to FP16
    model.half()
    # save model to disk
    model.save_pretrained(output_dir)

    # export also processor
    asr_processor = AutoProcessor.from_pretrained(MODEL_NAME)
    asr_processor.save_pretrained(output_dir)

    return Path(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")

    args = parser.parse_args()
    convert_asr_model(Path(args.model_dir))