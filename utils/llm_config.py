DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

DEFAULT_SYSTEM_PROMPT_CHINESE = """\
你是一个乐于助人、尊重他人以及诚实可靠的助手。在安全的情况下，始终尽可能有帮助地回答。 您的回答不应包含任何有害、不道德、种族主义、性别歧视、有毒、危险或非法的内容。请确保您的回答在社会上是公正的和积极的。
如果一个问题没有任何意义或与事实不符，请解释原因，而不是回答错误的问题。如果您不知道问题的答案，请不要分享虚假信息。另外，答案请使用中文。\
"""

DEFAULT_SYSTEM_PROMPT_JAPANESE = """\
あなたは親切で、礼儀正しく、誠実なアシスタントです。 常に安全を保ちながら、できるだけ役立つように答えてください。 回答には、有害、非倫理的、人種差別的、性差別的、有毒、危険、または違法なコンテンツを含めてはいけません。 回答は社会的に偏見がなく、本質的に前向きなものであることを確認してください。
質問が意味をなさない場合、または事実に一貫性がない場合は、正しくないことに答えるのではなく、その理由を説明してください。 質問の答えがわからない場合は、誤った情報を共有しないでください。\
"""

DEFAULT_RAG_PROMPT = """\
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
"""

DEFAULT_RAG_PROMPT_CHINESE = """\
基于以下已知信息，请简洁并专业地回答用户的问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。\
"""


def red_pijama_partial_text_processor(partial_text, new_text):
    if new_text == "<":
        return partial_text

    partial_text += new_text
    return partial_text.split("<bot>:")[-1]


def deepseek_partial_text_processor(partial_text, new_text):
    partial_text += new_text
    return partial_text.split("</think>")[-1]


def llama_partial_text_processor(partial_text, new_text):
    new_text = new_text.replace("[INST]", "").replace("[/INST]", "")
    partial_text += new_text
    return partial_text


def chatglm_partial_text_processor(partial_text, new_text):
    new_text = new_text.strip()
    new_text = new_text.replace("[[训练时间]]", "2023年")
    partial_text += new_text
    return partial_text


def youri_partial_text_processor(partial_text, new_text):
    new_text = new_text.replace("システム:", "")
    partial_text += new_text
    return partial_text


def internlm_partial_text_processor(partial_text, new_text):
    partial_text += new_text
    return partial_text.split("<|im_end|>")[0]


def phi_completion_to_prompt(completion):
    return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"


def llama3_completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def qwen_completion_to_prompt(completion):
    return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"


SUPPORTED_LLM_MODELS = {
    "English": {
        "qwen2.5-0.5b-instruct": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT,
            "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            "completion_to_prompt": qwen_completion_to_prompt,
        },
        "tiny-llama-1b-chat": {
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "remote_code": False,
            "start_message": f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}</s>\n",
            "history_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}</s> \n",
            "current_message_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}",
            "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }</s>"""
            + """
            <|user|>
            Question: {input} 
            Context: {context} 
            Answer: </s>
            <|assistant|>""",
        },
        "DeepSeek-R1-Distill-Qwen-1.5B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assistant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT + "Think briefly and provide informative answers, avoidi mixing languages.",
        },
        "DeepSeek-R1-Distill-Qwen-7B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assistant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT + "Think briefly and provide informative answers, avoid mixing languages.",
        },
        "DeepSeek-R1-Distill-Llama-8B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assistant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT + "Think briefly and provide informative answers, avoid mixing languages.",
        },
        "llama-3.2-1b-instruct": {
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "start_message": DEFAULT_SYSTEM_PROMPT,
            "stop_tokens": ["<|eot_id|>"],
            "has_chat_template": True,
            "start_message": " <|start_header_id|>system<|end_header_id|>\n\n" + DEFAULT_SYSTEM_PROMPT + "<|eot_id|>",
            "history_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
            "current_message_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}",
            "rag_prompt_template": f"<|start_header_id|>system<|end_header_id|>\n\n{DEFAULT_RAG_PROMPT}<|eot_id|>"
            + """<|start_header_id|>user<|end_header_id|>
            
            
            Question: {input}
            Context: {context}
            Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            
            """,
            "completion_to_prompt": llama3_completion_to_prompt,
        },
        "llama-3.2-3b-instruct": {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "start_message": DEFAULT_SYSTEM_PROMPT,
            "stop_tokens": ["<|eot_id|>"],
            "has_chat_template": True,
            "start_message": " <|start_header_id|>system<|end_header_id|>\n\n" + DEFAULT_SYSTEM_PROMPT + "<|eot_id|>",
            "history_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
            "current_message_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}",
            "rag_prompt_template": f"<|start_header_id|>system<|end_header_id|>\n\n{DEFAULT_RAG_PROMPT}<|eot_id|>"
            + """<|start_header_id|>user<|end_header_id|>
            
            
            Question: {input}
            Context: {context}
            Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            
            """,
            "completion_to_prompt": llama3_completion_to_prompt,
        },
        "qwen2.5-1.5b-instruct": {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT,
            "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            "completion_to_prompt": qwen_completion_to_prompt,
        },
        "gemma-2b-it": {
            "model_id": "google/gemma-2b-it",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT + ", ",
            "history_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}<end_of_turn>",
            "current_message_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}",
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT},"""
            + """<start_of_turn>user{input}<end_of_turn><start_of_turn>context{context}<end_of_turn><start_of_turn>model""",
        },
        "gemma-2-2b-it": {
            "model_id": "google/gemma-2-2b-it",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT + ", ",
            "history_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}<end_of_turn>",
            "current_message_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}",
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT},"""
            + """<start_of_turn>user{input}<end_of_turn><start_of_turn>context{context}<end_of_turn><start_of_turn>model""",
        },
        "red-pajama-3b-chat": {
            "model_id": "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
            "remote_code": False,
            "start_message": "",
            "history_template": "\n<human>:{user}\n<bot>:{assistant}",
            "stop_tokens": [29, 0],
            "partial_text_processor": red_pijama_partial_text_processor,
            "current_message_template": "\n<human>:{user}\n<bot>:{assistant}",
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT }"""
            + """
            <human>: Question: {input} 
            Context: {context} 
            Answer: <bot>""",
        },
        "qwen2.5-3b-instruct": {
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT + ", ",
            "rag_prompt_template": f"""<|im_start|>system
            {DEFAULT_RAG_PROMPT }<|im_end|>"""
            + """
            <|im_start|>user
            Question: {input} 
            Context: {context} 
            Answer: <|im_end|>
            <|im_start|>assistant
            """,
            "completion_to_prompt": qwen_completion_to_prompt,
        },
        "minicpm3-4b": {"model_id": "openbmb/MiniCPM3-4B", "remote_code": True, "start_message": DEFAULT_SYSTEM_PROMPT},
        "qwen2.5-7b-instruct": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT + ", ",
            "rag_prompt_template": f"""<|im_start|>system
            {DEFAULT_RAG_PROMPT }<|im_end|>"""
            + """
            <|im_start|>user
            Question: {input} 
            Context: {context} 
            Answer: <|im_end|>
            <|im_start|>assistant
            """,
            "completion_to_prompt": qwen_completion_to_prompt,
        },
        "gemma-7b-it": {
            "model_id": "google/gemma-7b-it",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT + ", ",
            "history_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}<end_of_turn>",
            "current_message_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}",
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT},"""
            + """<start_of_turn>user{input}<end_of_turn><start_of_turn>context{context}<end_of_turn><start_of_turn>model""",
        },
        "gemma-2-9b-it": {
            "model_id": "google/gemma-2-9b-it",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT + ", ",
            "history_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}<end_of_turn>",
            "current_message_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}",
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT},"""
            + """<start_of_turn>user{input}<end_of_turn><start_of_turn>context{context}<end_of_turn><start_of_turn>model""",
        },
        "llama-2-chat-7b": {
            "model_id": "meta-llama/Llama-2-7b-chat-hf",
            "remote_code": False,
            "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n",
            "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
            "current_message_template": "{user} [/INST]{assistant}",
            "tokenizer_kwargs": {"add_special_tokens": False},
            "partial_text_processor": llama_partial_text_processor,
            "rag_prompt_template": f"""[INST]Human: <<SYS>> {DEFAULT_RAG_PROMPT }<</SYS>>"""
            + """
            Question: {input} 
            Context: {context} 
            Answer: [/INST]""",
        },
        "llama-3-8b-instruct": {
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT,
            "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
            "has_chat_template": True,
            "start_message": " <|start_header_id|>system<|end_header_id|>\n\n" + DEFAULT_SYSTEM_PROMPT + "<|eot_id|>",
            "history_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
            "current_message_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}",
            "rag_prompt_template": f"<|start_header_id|>system<|end_header_id|>\n\n{DEFAULT_RAG_PROMPT}<|eot_id|>"
            + """<|start_header_id|>user<|end_header_id|>
            
            
            Question: {input}
            Context: {context}
            Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            
            """,
            "completion_to_prompt": llama3_completion_to_prompt,
        },
        "llama-3.1-8b-instruct": {
            "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT,
            "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
            "has_chat_template": True,
            "start_message": " <|start_header_id|>system<|end_header_id|>\n\n" + DEFAULT_SYSTEM_PROMPT + "<|eot_id|>",
            "history_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
            "current_message_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}",
            "rag_prompt_template": f"<|start_header_id|>system<|end_header_id|>\n\n{DEFAULT_RAG_PROMPT}<|eot_id|>"
            + """<|start_header_id|>user<|end_header_id|>
            
            
            Question: {input}
            Context: {context}
            Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            
            """,
            "completion_to_prompt": llama3_completion_to_prompt,
        },
        "mistral-7b-instruct": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
            "remote_code": False,
            "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n",
            "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
            "current_message_template": "{user} [/INST]{assistant}",
            "tokenizer_kwargs": {"add_special_tokens": False},
            "partial_text_processor": llama_partial_text_processor,
            "rag_prompt_template": f"""<s> [INST] {DEFAULT_RAG_PROMPT } [/INST] </s>"""
            + """ 
            [INST] Question: {input} 
            Context: {context} 
            Answer: [/INST]""",
        },
        "mistral-7B-Instruct-v0.3": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "remote_code": False,
            "start_message": f"<s>[INST] {DEFAULT_SYSTEM_PROMPT }\n\n",
            "history_template": "{user}[/INST]{assistant}</s>[INST]",
            "current_message_template": "{user} [/INST]{assistant}</s>",
            "tokenizer_kwargs": {"add_special_tokens": False},
            "partial_text_processor": llama_partial_text_processor,
            "genai_chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{% if (messages[0]['role'] == 'system' and messages|length == 2) %}{{ message['content'] + '[/INST]' }}{% else %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% endif %}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token + ' ' }}{% elif (message['role'] == 'system' and messages|length == 2) %}{{ '[INST] ' + message['content'] + ' \n\n' }}{% else %}{{ raise_exception('Only system, user and assistant roles are supported!') }}{% endif %}{% endfor %}",
        },
        "zephyr-7b-beta": {
            "model_id": "HuggingFaceH4/zephyr-7b-beta",
            "remote_code": False,
            "start_message": f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}</s>\n",
            "history_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}</s> \n",
            "current_message_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}",
            "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }</s>"""
            + """ 
            <|user|>
            Question: {input} 
            Context: {context} 
            Answer: </s>
            <|assistant|>""",
        },
        "notus-7b-v1": {
            "model_id": "argilla/notus-7b-v1",
            "remote_code": False,
            "start_message": f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}</s>\n",
            "history_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}</s> \n",
            "current_message_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}",
            "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }</s>"""
            + """
            <|user|>
            Question: {input} 
            Context: {context} 
            Answer: </s>
            <|assistant|>""",
        },
        "neural-chat-7b-v3-3": {
            "model_id": "Intel/neural-chat-7b-v3-3",
            "remote_code": False,
            "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n",
            "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
            "current_message_template": "{user} [/INST]{assistant}",
            "tokenizer_kwargs": {"add_special_tokens": False},
            "partial_text_processor": llama_partial_text_processor,
            "rag_prompt_template": f"""<s> [INST] {DEFAULT_RAG_PROMPT } [/INST] </s>"""
            + """
            [INST] Question: {input} 
            Context: {context} 
            Answer: [/INST]""",
        },
        "phi-3-mini-instruct": {
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "remote_code": True,
            "start_message": "<|system|>\n{DEFAULT_SYSTEM_PROMPT}<|end|>\n",
            "history_template": "<|user|>\n{user}<|end|> \n<|assistant|>\n{assistant}<|end|>\n",
            "current_message_template": "<|user|>\n{user}<|end|> \n<|assistant|>\n{assistant}",
            "stop_tokens": ["<|end|>"],
            "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }<|end|>"""
            + """
            <|user|>
            Question: {input} 
            Context: {context} 
            Answer: <|end|>
            <|assistant|>""",
            "completion_to_prompt": phi_completion_to_prompt,
        },
        "phi-3.5-mini-instruct": {
            "model_id": "microsoft/Phi-3.5-mini-instruct",
            "remote_code": True,
            "start_message": "<|system|>\n{DEFAULT_SYSTEM_PROMPT}<|end|>\n",
            "history_template": "<|user|>\n{user}<|end|> \n<|assistant|>\n{assistant}<|end|>\n",
            "current_message_template": "<|user|>\n{user}<|end|> \n<|assistant|>\n{assistant}",
            "stop_tokens": ["<|end|>"],
            "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }<|end|>"""
            + """
            <|user|>
            Question: {input} 
            Context: {context} 
            Answer: <|end|>
            <|assistant|>""",
            "completion_to_prompt": phi_completion_to_prompt,
        },
        "phi-4-mini-instruct": {"model_id": "microsoft/phi-4-mini-instruct", "remote_code": True, "start_message": DEFAULT_SYSTEM_PROMPT},
        "phi-4": {"model_id": "microsoft/phi-4", "remote_code": False, "start_message": DEFAULT_SYSTEM_PROMPT},
        "qwen2.5-14b-instruct": {
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT + ", ",
            "rag_prompt_template": f"""<|im_start|>system
            {DEFAULT_RAG_PROMPT }<|im_end|>"""
            + """
            <|im_start|>user
            Question: {input} 
            Context: {context} 
            Answer: <|im_end|>
            <|im_start|>assistant
            """,
            "completion_to_prompt": qwen_completion_to_prompt,
        },
    },
    "Chinese": {
        "qwen2.5-0.5b-instruct": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            "completion_to_prompt": qwen_completion_to_prompt,
        },
        "qwen2.5-1.5b-instruct": {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            "completion_to_prompt": qwen_completion_to_prompt,
        },
        "qwen2.5-3b-instruct": {
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            "completion_to_prompt": qwen_completion_to_prompt,
        },
        "qwen2.5-7b-instruct": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "remote_code": False,
            "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "rag_prompt_template": f"""<|im_start|>system
            {DEFAULT_RAG_PROMPT_CHINESE }<|im_end|>"""
            + """
            <|im_start|>user
            问题: {input} 
            已知内容: {context} 
            回答: <|im_end|>
            <|im_start|>assistant
            """,
            "completion_to_prompt": qwen_completion_to_prompt,
        },
        "qwen2.5-14b-instruct": {
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "remote_code": False,
            "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "rag_prompt_template": f"""<|im_start|>system
            {DEFAULT_RAG_PROMPT_CHINESE }<|im_end|>"""
            + """
            <|im_start|>user
            问题: {input} 
            已知内容: {context} 
            回答: <|im_end|>
            <|im_start|>assistant
            """,
            "completion_to_prompt": qwen_completion_to_prompt,
        },
        "qwen-7b-chat": {
            "model_id": "Qwen/Qwen-7B-Chat",
            "remote_code": True,
            "start_message": f"<|im_start|>system\n {DEFAULT_SYSTEM_PROMPT_CHINESE }<|im_end|>",
            "history_template": "<|im_start|>user\n{user}<im_end><|im_start|>assistant\n{assistant}<|im_end|>",
            "current_message_template": '"<|im_start|>user\n{user}<im_end><|im_start|>assistant\n{assistant}',
            "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            "revision": "2abd8e5777bb4ce9c8ab4be7dbbd0fe4526db78d",
            "rag_prompt_template": f"""<|im_start|>system
            {DEFAULT_RAG_PROMPT_CHINESE }<|im_end|>"""
            + """
            <|im_start|>user
            问题: {input} 
            已知内容: {context} 
            回答: <|im_end|>
            <|im_start|>assistant
            """,
        },
        "chatglm3-6b": {
            "model_id": "THUDM/chatglm3-6b",
            "remote_code": True,
            "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "tokenizer_kwargs": {"add_special_tokens": False},
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT_CHINESE }"""
            + """
            问题: {input} 
            已知内容: {context} 
            回答: 
            """,
        },
        "glm-4-9b-chat": {
            "model_id": "THUDM/glm-4-9b-chat",
            "remote_code": True,
            "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "tokenizer_kwargs": {"add_special_tokens": False},
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT_CHINESE }"""
            + """
            问题: {input} 
            已知内容: {context} 
            回答: 
            """,
        },
        "baichuan2-7b-chat": {
            "model_id": "baichuan-inc/Baichuan2-7B-Chat",
            "remote_code": True,
            "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "tokenizer_kwargs": {"add_special_tokens": False},
            "stop_tokens": ["<unk>", "</s>"],
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT_CHINESE }"""
            + """
            问题: {input} 
            已知内容: {context} 
            回答: 
            """,
        },
        "minicpm-2b-dpo": {
            "model_id": "openbmb/MiniCPM-2B-dpo-fp16",
            "remote_code": True,
            "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
        },
        "minicpm3-4b": {"model_id": "openbmb/MiniCPM3-4B", "remote_code": True, "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE},
        "internlm2-chat-1.8b": {
            "model_id": "internlm/internlm2-chat-1_8b",
            "remote_code": True,
            "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "stop_tokens": ["</s>", "<|im_end|>"],
            "partial_text_processor": internlm_partial_text_processor,
        },
        "qwen1.5-1.8b-chat": {
            "model_id": "Qwen/Qwen1.5-1.8B-Chat",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            "rag_prompt_template": f"""<|im_start|>system
            {DEFAULT_RAG_PROMPT_CHINESE }<|im_end|>"""
            + """
            <|im_start|>user
            问题: {input} 
            已知内容: {context} 
            回答: <|im_end|>
            <|im_start|>assistant
            """,
        },
    },
    "Japanese": {
        "youri-7b-chat": {
            "model_id": "rinna/youri-7b-chat",
            "remote_code": False,
            "start_message": f"設定: {DEFAULT_SYSTEM_PROMPT_JAPANESE}\n",
            "history_template": "ユーザー: {user}\nシステム: {assistant}\n",
            "current_message_template": "ユーザー: {user}\nシステム: {assistant}",
            "tokenizer_kwargs": {"add_special_tokens": False},
            "partial_text_processor": youri_partial_text_processor,
        },
    },
}

SUPPORTED_EMBEDDING_MODELS = {
    "English": {
        "bge-small-en-v1.5": {
            "model_id": "BAAI/bge-small-en-v1.5",
            "mean_pooling": False,
            "normalize_embeddings": True,
        },
        "bge-large-en-v1.5": {
            "model_id": "BAAI/bge-large-en-v1.5",
            "mean_pooling": False,
            "normalize_embeddings": True,
        },
        "bge-m3": {
            "model_id": "BAAI/bge-m3",
            "mean_pooling": False,
            "normalize_embeddings": True,
        },
    },
    "Chinese": {
        "bge-small-zh-v1.5": {
            "model_id": "BAAI/bge-small-zh-v1.5",
            "mean_pooling": False,
            "normalize_embeddings": True,
        },
        "bge-large-zh-v1.5": {
            "model_id": "BAAI/bge-large-zh-v1.5",
            "mean_pooling": False,
            "normalize_embeddings": True,
        },
        "bge-m3": {
            "model_id": "BAAI/bge-m3",
            "mean_pooling": False,
            "normalize_embeddings": True,
        },
    },
}


SUPPORTED_RERANK_MODELS = {
    "bge-reranker-v2-m3": {"model_id": "BAAI/bge-reranker-v2-m3"},
    "bge-reranker-large": {"model_id": "BAAI/bge-reranker-large"},
    "bge-reranker-base": {"model_id": "BAAI/bge-reranker-base"},
}

compression_configs = {
    "zephyr-7b-beta": {
        "sym": True,
        "group_size": 64,
        "ratio": 0.6,
    },
    "mistral-7b": {
        "sym": True,
        "group_size": 64,
        "ratio": 0.6,
    },
    "minicpm-2b-dpo": {
        "sym": True,
        "group_size": 64,
        "ratio": 0.6,
    },
    "gemma-2b-it": {
        "sym": True,
        "group_size": 64,
        "ratio": 0.6,
    },
    "notus-7b-v1": {
        "sym": True,
        "group_size": 64,
        "ratio": 0.6,
    },
    "neural-chat-7b-v3-1": {
        "sym": True,
        "group_size": 64,
        "ratio": 0.6,
    },
    "llama-2-chat-7b": {
        "sym": True,
        "group_size": 128,
        "ratio": 0.8,
    },
    "llama-3-8b-instruct": {
        "sym": True,
        "group_size": 128,
        "ratio": 0.8,
    },
    "gemma-7b-it": {
        "sym": True,
        "group_size": 128,
        "ratio": 0.8,
    },
    "chatglm2-6b": {
        "sym": True,
        "group_size": 128,
        "ratio": 0.72,
    },
    "qwen-7b-chat": {"sym": True, "group_size": 128, "ratio": 0.6},
    "qwen2.5-7b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
    "qwen2.5-3b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
    "qwen2.5-14b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
    "qwen2.5-1.5b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
    "qwen2.5-0.5b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
    "red-pajama-3b-chat": {
        "sym": False,
        "group_size": 128,
        "ratio": 0.5,
    },
    "llama-3.2-3b-instruct": {"sym": False, "group_size": 64, "ratio": 1.0, "dataset": "wikitext2", "awq": True, "all_layers": True, "scale_estimation": True},
    "llama-3.2-1b-instruct": {"sym": False, "group_size": 64, "ratio": 1.0, "dataset": "wikitext2", "awq": True, "all_layers": True, "scale_estimation": True},
    "default": {
        "sym": False,
        "group_size": 128,
        "ratio": 0.8,
    },
}


def get_optimum_cli_command(model_id, weight_format, output_dir, compression_options=None, enable_awq=False, trust_remote_code=False):
    base_command = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format {}"
    command = base_command.format(model_id, weight_format)
    if compression_options:
        compression_args = " --group-size {} --ratio {}".format(compression_options["group_size"], compression_options["ratio"])
        if compression_options["sym"]:
            compression_args += " --sym"
        if enable_awq or compression_options.get("awq", False):
            compression_args += " --awq --dataset wikitext2 --num-samples 128"
            if compression_options.get("scale_estimation", False):
                compression_args += " --scale-estimation"
        if compression_options.get("all_layers", False):
            compression_args += " --all-layers"

        command = command + compression_args
    if trust_remote_code:
        command += " --trust-remote-code"

    command += " {}".format(output_dir)
    return command


default_language = "English"

SUPPORTED_OPTIMIZATIONS = ["INT4", "INT4-AWQ", "INT4-NPU", "INT8", "FP16"]

int4_npu_config = {
    "sym": True,
    "group_size": -1,
    "ratio": 1.0,
}


def get_llm_selection_widget(languages=list(SUPPORTED_LLM_MODELS), models=SUPPORTED_LLM_MODELS[default_language], show_preconverted_checkbox=True, device=None):
    import ipywidgets as widgets

    lang_dropdown = widgets.Dropdown(options=languages or [])

    # Define dependent drop down

    model_dropdown = widgets.Dropdown(options=models)

    def dropdown_handler(change):
        global default_language
        default_language = change.new
        # If statement checking on dropdown value and changing options of the dependent dropdown accordingly
        model_dropdown.options = SUPPORTED_LLM_MODELS[change.new]

    lang_dropdown.observe(dropdown_handler, names="value")
    compression_dropdown = widgets.Dropdown(options=SUPPORTED_OPTIMIZATIONS if device != "NPU" else ["INT4-NPU", "FP16"])
    preconverted_checkbox = widgets.Checkbox(value=True)

    form_items = []

    if languages:
        form_items.append(widgets.Box([widgets.Label(value="Language:"), lang_dropdown]))
    form_items.extend(
        [
            widgets.Box([widgets.Label(value="Model:"), model_dropdown]),
            widgets.Box([widgets.Label(value="Compression:"), compression_dropdown]),
        ]
    )
    if show_preconverted_checkbox:
        form_items.append(widgets.Box([widgets.Label(value="Use preconverted models:"), preconverted_checkbox]))

    form = widgets.Box(
        form_items,
        layout=widgets.Layout(
            display="flex",
            flex_flow="column",
            border="solid 1px",
            # align_items='stretch',
            width="30%",
            padding="1%",
        ),
    )
    return form, lang_dropdown, model_dropdown, compression_dropdown, preconverted_checkbox


def convert_tokenizer(model_id, remote_code, model_dir):
    import openvino as ov
    from transformers import AutoTokenizer
    from openvino_tokenizers import convert_tokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=remote_code)
    ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, model_dir / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, model_dir / "openvino_detokenizer.xml")


def convert_and_compress_model(model_id, model_config, precision, use_preconverted=True):
    from pathlib import Path
    from IPython.display import Markdown, display
    import subprocess  # nosec - disable B404:import-subprocess check
    import platform

    pt_model_id = model_config["model_id"]
    pt_model_name = model_id.split("/")[-1]
    model_subdir = precision if precision == "FP16" else precision + "_compressed_weights"
    model_dir = Path(pt_model_name) / model_subdir
    remote_code = model_config.get("remote_code", False)
    if (model_dir / "openvino_model.xml").exists():
        print(f"✅ {precision} {model_id} model already converted and can be found in {model_dir}")

        if not (model_dir / "openvino_tokenizer.xml").exists() or not (model_dir / "openvino_detokenizer.xml").exists():
            convert_tokenizer(pt_model_id, remote_code, model_dir)
        return model_dir
    if use_preconverted:
        OV_ORG = "OpenVINO"
        pt_model_name = pt_model_id.split("/")[-1]
        ov_model_name = pt_model_name + f"-{precision.lower()}-ov"
        ov_model_hub_id = f"{OV_ORG}/{ov_model_name}"
        import huggingface_hub as hf_hub

        hub_api = hf_hub.HfApi()
        if hub_api.repo_exists(ov_model_hub_id):
            print(f"⌛Found preconverted {precision} {model_id}. Downloading model started. It may takes some time.")
            hf_hub.snapshot_download(ov_model_hub_id, local_dir=model_dir)
            print(f"✅ {precision} {model_id} model downloaded and can be found in {model_dir}")
            return model_dir

    model_compression_params = {}
    if "INT4" in precision:
        model_compression_params = compression_configs.get(model_id, compression_configs["default"]) if not "NPU" in precision else int4_npu_config
    weight_format = precision.split("-")[0].lower()
    optimum_cli_command = get_optimum_cli_command(pt_model_id, weight_format, model_dir, model_compression_params, "AWQ" in precision, remote_code)
    print(f"⌛ {model_id} conversion to {precision} started. It may takes some time.")
    display(Markdown("**Export command:**"))
    display(Markdown(f"`{optimum_cli_command}`"))
    subprocess.run(optimum_cli_command.split(" "), shell=(platform.system() == "Windows"), check=True)
    print(f"✅ {precision} {model_id} model converted and can be found in {model_dir}")
    return model_dir


def compare_model_size(model_dir):
    fp16_weights = model_dir.parent / "FP16" / "openvino_model.bin"
    int8_weights = model_dir.parent / "INT8_compressed_weights" / "openvino_model.bin"
    int4_weights = model_dir.parent / "INT4_compressed_weights" / "openvino_model.bin"
    int4_awq_weights = model_dir.parent / "INT4-AWQ_compressed_weights" / "openvino_model.bin"
    int4_npu_weights = model_dir.parent / "INT4-NPU_compressed_weights" / "openvino_model.bin"

    if fp16_weights.exists():
        print(f"Size of FP16 model is {fp16_weights.stat().st_size / 1024 / 1024:.2f} MB")
    for precision, compressed_weights in zip(["INT8", "INT4", "INT4-AWQ", "INT4-NPU"], [int8_weights, int4_weights, int4_awq_weights, int4_npu_weights]):
        if compressed_weights.exists():
            print(f"Size of model with {precision} compressed weights is {compressed_weights.stat().st_size / 1024 / 1024:.2f} MB")
        if compressed_weights.exists() and fp16_weights.exists():
            print(f"Compression rate for {precision} model: {fp16_weights.stat().st_size / compressed_weights.stat().st_size:.3f}")
