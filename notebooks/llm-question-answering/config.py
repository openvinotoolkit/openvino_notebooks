SUPPORTED_LLM_MODELS = {
    "tiny-llama-1b": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt_template": "<|user|>\n{instruction}</s> \n<|assistant|>\n",
        "tokenizer_kwargs": {"add_special_tokens": False},
    },
    "phi-2": {
        "model_id": "susnato/phi-2",
        "prompt_template": "Instruct:{instruction}\nOutput:",
    },
    "red-pajama-instruct-3b": {
        "model_id": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        "prompt_template": "Q: {instruction}\nA:",
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "prompt_template": "<s> [INST] {instruction} [/INST] </s>",
        "tokenizer_kwargs": {"add_special_tokens": False},
    },
    "llama-3-8b-instruct": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "end_key": "<|eot_id|>",
        "prompt_template": "<|start_header_id|>system<|end_header_id|>\n\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.<|eot_id|><|start_header_id|>user<|end_header_id|>Instruction: {instruction} Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    },
    "Qwen2.5-0.5B-Instruct-GGUF": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "gguf_file": ["qwen2.5-0.5b-instruct-fp16.gguf", "qwen2.5-0.5b-instruct-q4_0.gguf", "qwen2.5-0.5b-instruct-q8_0.gguf"],
    },
    "SmolLM2-135M-GGUF": {"model_id": "prithivMLmods/SmolLM2-135M-GGUF", "gguf_file": ["SmolLM2-135M.F16.gguf", "SmolLM2-135M.Q8_0.gguf"]},
}
