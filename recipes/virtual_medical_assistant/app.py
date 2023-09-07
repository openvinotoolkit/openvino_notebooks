import argparse

from optimum.intel import OVModelForCausalLM
from transformers import AutoConfig, AutoTokenizer


def run(chat_model_dir):
    system_prompt = """You are a helpful, respectful and honest doctor assistant. 
    Your role is talking to a patient, who just came in. 
    Your task is to get all symptoms from the patient and suggest to the doctor a plausible problem. 
    You cannot treat the patient yourself."""

    # template specific to Red Pajama model
    message_template = "\n<human>: {}\n<bot>: {}"

    prompt = system_prompt + message_template.format("I feel pain in my back", "")

    chat_model = OVModelForCausalLM.from_pretrained(chat_model_dir, device="AUTO", config=AutoConfig.from_pretrained(chat_model_dir))
    tokenizer = AutoTokenizer.from_pretrained(chat_model_dir)

    inputs = tokenizer(prompt, return_tensors="pt").to(chat_model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = chat_model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50)
    # token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(outputs[0])
    print(output_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat_model_dir', type=str, default="model/red_pajama", help="Path to the chat model directory")

    args = parser.parse_args()
    run(args.chat_model_dir)
