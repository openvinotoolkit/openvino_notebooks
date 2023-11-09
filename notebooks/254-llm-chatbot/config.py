from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""


def red_pijama_partial_text_processor(partial_text, new_text):
    if new_text == '<':
        return partial_text
    
    partial_text += new_text
    return partial_text.split('<bot>:')[-1]

def llama_partial_text_processor(partial_text, new_text):
    new_text = new_text.replace("[INST]", "").replace("[/INST]", "")
    partial_text += new_text
    return partial_text

SUPPORTED_MODELS = {
  "red-pajama-3b-chat" : {"model_id": "togethercomputer/RedPajama-INCITE-Chat-3B-v1", "start_message": "", "history_template": "\n<human>:{user}\n<bot>:{assistant}", "stop_tokens": [29, 0], "partial_text_processor": red_pijama_partial_text_processor, "current_message_template": "\n<human>:{user}\n<bot>:{assistant}"},
  "llama-2-chat-7b": {"model_id": "meta-llama/Llama-2-7b-chat-hf", "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n", "history_template": "{user}[/INST]{assistant}</s><s>[INST]", "current_message_template": "{user} [/INST]{assistant}","tokenizer_kwargs": {"add_special_tokens":False}, "partial_text_processor": llama_partial_text_processor, "revision": "5514c85fedd6c4fc0fc66fa533bc157de520da73"},
  "mpt-7b-chat": {"model_id": "mosaicml/mpt-7b-chat", "start_message": f"<|im_start|>system\n {DEFAULT_SYSTEM_PROMPT }<|im_end|>", "history_template": "<|im_start|>user\n{user}<im_end><|im_start|>assistant\n{assistant}<|im_end|>", "current_message_template": "\"<|im_start|>user\n{user}<im_end><|im_start|>assistant\n{assistant}", "stop_tokens": ["<|im_end|>", "<|endoftext|>"]},
"zephyr-7b-beta": {"model_id": "HuggingFaceH4/zephyr-7b-beta", "start_message":f'<|system|>\n{DEFAULT_SYSTEM_PROMPT}</s>\n', "history_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}</s> \n",  "current_message_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}"}
}
