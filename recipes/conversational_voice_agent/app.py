import argparse
from pathlib import Path

import gradio as gr
from optimum.intel import OVModelForCausalLM
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

CHAT_MODEL_TEMPLATES = {
    "llama2": {
        "system_configuration": "<<SYS>>\nYou're Adrishuo - a conversational agent. You talk to a customer. Your task is to recommend the customer products based on their needs.\n<</SYS>>\n\n",
        "message_template": "[INST] {} [/INST] {}\n"
    }
}

chat_model: OVModelForCausalLM = None
chat_tokenizer: PreTrainedTokenizer = None
system_configuration: str = None
message_template: str = None


def load_chat_model(model_dir: Path) -> None:
    """
    Load chat model and assign it to a global variable

    Params:
        model_dir: dir with the chat model
    """
    global chat_model, chat_tokenizer, system_configuration, message_template

    model_name = model_dir.name.split("-")[0]
    # system and message templates specific to the loaded model
    system_configuration = CHAT_MODEL_TEMPLATES[model_name]["system_configuration"]
    message_template = CHAT_MODEL_TEMPLATES[model_name]["message_template"]

    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    chat_model = OVModelForCausalLM.from_pretrained(model_dir, device="AUTO", config=AutoConfig.from_pretrained(model_dir), ov_config=ov_config)
    chat_tokenizer = AutoTokenizer.from_pretrained(model_dir)


def respond(prompt: str) -> str:
    """
    Respond to the current prompt

    Params:
        prompt: user's prompt
    Returns:
       The chat's response
    """
    inputs = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = chat_model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9, top_k=50)
    token = outputs[0, input_length:]
    return chat_tokenizer.decode(token).split("</s>")[0]


def chat(user_prompt: str, history: list) -> str:
    messages = [message_template.format(human, ai if ai is not None else "") for human, ai in history]
    messages.append(message_template.format(user_prompt, ""))
    conversation = system_configuration + "\n".join(messages)

    return respond(conversation)


def run(chat_model_dir: Path, public_interface: bool = False) -> None:
    """
    Run the assistant application

    Params
        chat_model_dir: dir with the chat model
        public_interface: whether UI should be available publicly
    """
    # load chat model
    load_chat_model(chat_model_dir)

    history = []
    initial_prompt = "Please introduce yourself and greet the customer"
    initial_message = chat(initial_prompt, history)

    chat_ui = gr.Chatbot(value=[(None, initial_message)])
    chatbot_ui = gr.ChatInterface(fn=chat, chatbot=chat_ui, title="Talk to Adrishuo - a conversational voice agent")

    chatbot_ui.launch(share=public_interface)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat_model_dir', type=str, default="model/llama2-7B", help="Path to the chat model directory")
    parser.add_argument('--public_interface', default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run(Path(args.chat_model_dir), args.public_interface)
