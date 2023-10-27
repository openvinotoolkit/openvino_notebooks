import argparse

from optimum.intel import OVModelForCausalLM
from transformers import AutoConfig, AutoTokenizer


def run(chat_model_dir):
    system_prompt = """<<SYS>>
You're an assistant of a doctor. You talk to a patient. Your task is to get all symptoms and suggest the problem to the doctor. Don't try to treat the patient yourself.
<</SYS>>

"""

    # {'content': '<<SYS>>\nAlways answer with Haiku\n<</SYS>>\n\nI am going to Paris, what should I see?', 'role': 'user'}

    # template specific to LLama2 model
    message_template = "[INST] {} [/INST] {}"

    prompt = system_prompt + message_template.format("I feel pain in my back", "")

    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    chat_model = OVModelForCausalLM.from_pretrained(chat_model_dir, device="CPU", config=AutoConfig.from_pretrained(chat_model_dir), ov_config=ov_config)
    tokenizer = AutoTokenizer.from_pretrained(chat_model_dir)

    inputs = tokenizer(prompt, return_tensors="pt").to(chat_model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = chat_model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9, top_k=50)
    # token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(outputs[0])
    print(output_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat_model_dir', type=str, default="model/llama2-7B", help="Path to the chat model directory")

    args = parser.parse_args()
    run(args.chat_model_dir)
