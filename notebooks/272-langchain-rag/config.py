def llama_partial_text_processor(partial_text, new_text):
    new_text = new_text.replace("[INST]", "").replace("[/INST]", "")
    partial_text += new_text
    return partial_text

SUPPORTED_EMBEDDING_MODELS = {"all-mpnet-base-v2": {"model_id": "sentence-transformers/all-mpnet-base-v2"}}

SUPPORTED_LLM_MODELS = {
    "llama-2-chat-7b": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "prompt_id": "rlm/rag-prompt-llama",
        "partial_text_processor": llama_partial_text_processor,
    },
    "mistal-7b": {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "prompt_id": "rlm/rag-prompt-mistral",
        "partial_text_processor": llama_partial_text_processor,
        
    },
    "neural-chat-7b-v3-1": {
        "model_id": "Intel/neural-chat-7b-v3-1",
        "prompt_id": "rlm/rag-prompt-mistral",
        "partial_text_processor": llama_partial_text_processor,
    },
}
