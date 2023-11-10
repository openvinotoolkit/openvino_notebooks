import argparse
from pathlib import Path

from optimum.intel import OVModelForCausalLM, OVQuantizer
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_MAPPING = {
    "llama2-7B": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13B": "meta-llama/Llama-2-13b-chat-hf"
}


def convert_chat_model(model_type: str, quantize_weights: bool, model_dir: Path) -> Path:
    """
    Convert chat model

    Params:
        model_type: selected mode type and size
        quantize_weights: whether quantize weights to INT8
        model_dir: dir to export model
    Returns:
       Path to exported model
    """
    output_dir = model_dir / model_type
    model_name = MODEL_MAPPING[model_type]

    if quantize_weights:
        # load model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # change precision to INT8 and save to disk
        quantizer = OVQuantizer.from_pretrained(model)
        output_dir = output_dir.with_name(output_dir.name + "-INT8")
        quantizer.quantize(save_directory=output_dir, weights_only=True)
    else:
        # load model and convert it to OpenVINO
        model = OVModelForCausalLM.from_pretrained(model_name, export=True, compile=False)
        # change precision to FP16
        model.half()
        # save model to disk
        output_dir = output_dir.with_name(output_dir.name + "-FP16")
        model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    return Path(output_dir) / "openvino_model.xml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model_type", type=str, choices=["llama2-7B", "llama2-13B"],
                        default="llama2-7B", help="Chat model to be converted")
    parser.add_argument("--quantize_weights", default=False, action="store_true", help="Whether to quantize weights to INT8")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")

    args = parser.parse_args()
    convert_chat_model(args.chat_model_type, args.quantize_weights, Path(args.model_dir))
