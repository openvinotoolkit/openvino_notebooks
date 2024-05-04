SUPPORTED_LLM_MODELS = {
    "tiny-llama-1b": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        "prompt_template":  "<|user|>\n{instruction}</s> \n<|assistant|>\n",
        "tokenizer_kwargs": {"add_special_tokens": False},
    },
    "phi-2": {
        "model_id": "susnato/phi-2",
        "prompt_template": "Instruct:{instruction}\nOutput:"
    },
    "dolly-v2-3b": {
        "model_id": "databricks/dolly-v2-3b",
        "instriction_key": "### Instruction:",
        "response_key": "### Response:",
        "end_key": "### End",
        "prompt_template": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    },
    "red-pajama-instruct-3b": {
        "model_id": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        "prompt_template": "Q: {instruction}\nA:"
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "prompt_template": "<s> [INST] {instruction} [/INST] </s>",
        "tokenizer_kwargs": {"add_special_tokens": False},
    }
}