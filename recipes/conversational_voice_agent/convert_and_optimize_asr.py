import argparse
from pathlib import Path

from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


def convert_asr_model(model_dir: Path) -> Path:
    """
    Convert speech-to-text model

    Params:
        model_dir: dir to export model
    Returns:
       Path to exported model
    """
     
    model_name = "distil-whisper/distil-large-v2"
    output_dir = model_dir / model_name.rsplit ("/")[-1]

    
    # load model and convert it to OpenVINO
    model = OVModelForSpeechSeq2Seq.from_pretrained(model_name, export=True, compile=False)
    # change precision to FP16
    model.half()
    # save model to disk
    output_dir = output_dir.with_name(output_dir.name + "-FP16")
    model.save_pretrained(output_dir)


    # export also processor
    asr_processor = AutoProcessor.from_pretrained(model_name)
    asr_processor.save_pretrained(output_dir)


    return Path(output_dir) / "openvino_model.xml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")

    args = parser.parse_args()
    convert_asr_model(Path(args.model_dir))