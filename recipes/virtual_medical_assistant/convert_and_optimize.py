import argparse
from pathlib import Path

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

RED_PAJAMA_MODEL_MAPPING = {
    "3B": "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
    "7B": "togethercomputer/RedPajama-INCITE-7B-Chat"
}


def convert_red_pajama(model_size, model_dir):
    """
    Convert Red Pajama Chat model

    Params:
        model_size: selected mode size
        model_dir: dir to export model
    Returns:
       Path to exported model
    """
    output_dir = model_dir / "red_pajama"
    model_name = RED_PAJAMA_MODEL_MAPPING[model_size]
    # load model and convert it to OpenVINO
    pajama_model = OVModelForCausalLM.from_pretrained(model_name, export=True, compile=False)
    # change precision to FP16
    pajama_model.half()
    # save model to disk
    pajama_model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    return Path(output_dir) / "openvino_model.xml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rp_model_size", type=str, choices=["3B", "7B"], default="3B",
                        help="Red Pajama model size to be converted")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")

    args = parser.parse_args()
    convert_red_pajama(args.rp_model_size, Path(args.model_dir))
