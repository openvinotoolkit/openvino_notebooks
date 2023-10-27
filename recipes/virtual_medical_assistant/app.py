import argparse
from pathlib import Path
from typing import Tuple

from optimum.intel import OVModelForCausalLM
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

CHAT_MODEL_TEMPLATES = {
    "llama2": {
        "system_configuration": "<<SYS>>\nYou're an assistant of a doctor called Adrishuo. You talk to a patient. Your task is to get all symptoms "
                                "and suggest the problem to the doctor. Don't try to treat the patient yourself.\n<</SYS>>\n\n",
        "message_template": "[INST] {} [/INST] {}\n"
    }
}


def load_chat_model(model_dir: Path) -> Tuple[OVModelForCausalLM, PreTrainedTokenizer]:
    """
    Load chat model

    Params:
        model_dir: dir with the chat model
    Returns:
       Chat model in OpenVINO and tokenizer
    """
    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    model = OVModelForCausalLM.from_pretrained(model_dir, device="AUTO", config=AutoConfig.from_pretrained(model_dir),
                                               ov_config=ov_config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def respond(prompt: str, model: OVModelForCausalLM, tokenizer: PreTrainedTokenizer) -> str:
    """
    Respond to the current prompt

    Params:
        prompt: user's prompt
        model: chat model
        tokenizer: tokenizer for the chat model
    Returns:
       The chat's response
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9, top_k=50)
    token = outputs[0, input_length:]
    return tokenizer.decode(token).split("</s>")[0]


def run(chat_model_dir: Path) -> None:
    """
    Run the assistant application

    Params:
        chat_model_dir: dir with the chat model
    """
    model_name = chat_model_dir.name.split("-")[0]
    # system and message templates specific to the loaded model
    system_configuration = CHAT_MODEL_TEMPLATES[model_name]["system_configuration"]
    message_template = CHAT_MODEL_TEMPLATES[model_name]["message_template"]

    # load chat model
    chat_model, chat_tokenizer = load_chat_model(chat_model_dir)

    history = []
    user_prompt = "Please introduce yourself and greet the patient"
    while True:
        messages = [message_template.format(human, ai) for human, ai in history]
        messages.append(message_template.format(user_prompt, ""))
        conversation = system_configuration + "\n".join(messages)

        output = respond(conversation, chat_model, chat_tokenizer)
        history.append((user_prompt, output))
        print(output)

        user_prompt = input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat_model_dir', type=str, default="model/llama2-7B", help="Path to the chat model directory")

    args = parser.parse_args()
    run(Path(args.chat_model_dir))
