DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

DEFAULT_SYSTEM_PROMPT_CHINESE = """\
你是一个乐于助人、尊重他人以及诚实可靠的助手。在安全的情况下，始终尽可能有帮助地回答。 您的回答不应包含任何有害、不道德、种族主义、性别歧视、有毒、危险或非法的内容。请确保您的回答在社会上是公正的和积极的。
如果一个问题没有任何意义或与事实不符，请解释原因，而不是回答错误的问题。如果您不知道问题的答案，请不要分享虚假信息。另外，答案请使用中文。\
"""


def deepseek_partial_text_processor(partial_text, new_text):
    partial_text += new_text
    return partial_text.split("</think>")[-1]


SUPPORTED_LLM_MODELS = {
    "English": {
        "DeepSeek-R1-Distill-Qwen-1.5B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "stop_strings": ["<｜end▁of▁sentence｜>", "<｜User｜>", "</User|>", "<|User|>", "<|end_of_sentence|>", "</｜"],
        },
        "DeepSeek-R1-Distill-Qwen-7B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "stop_strings": ["<｜end▁of▁sentence｜>", "<｜User｜>", "</User|>", "<|User|>", "<|end_of_sentence|>", "</｜"],
        },
        "DeepSeek-R1-Distill-Llama-8B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "stop_strings": ["<｜end▁of▁sentence｜>", "<｜User｜>", "</User|>", "<|User|>", "<|end_of_sentence|>", "</｜"],
        },
        "DeepSeek-R1-Distill-Qwen-14B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "stop_strings": ["<｜end▁of▁sentence｜>", "<｜User｜>", "</User|>", "<|User|>", "<|end_of_sentence|>", "</｜"],
        },
        "DeepSeek-R1-Distill-Qwen-32B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "stop_strings": ["<｜end▁of▁sentence｜>", "<｜User｜>", "</User|>", "<|User|>", "<|end_of_sentence|>", "</｜"],
        },
    },
    "Chinese": {
        "DeepSeek-R1-Distill-Qwen-1.5B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "stop_strings": ["<｜end▁of▁sentence｜>", "<｜User｜>", "</User|>", "<|User|>", "<|end_of_sentence|>", "</｜"],
        },
        "DeepSeek-R1-Distill-Qwen-7B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "stop_strings": ["<｜end▁of▁sentence｜>", "<｜User｜>", "</User|>", "<|User|>", "<|end_of_sentence|>", "</｜"],
        },
        "DeepSeek-R1-Distill-Llama-8B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "stop_strings": ["<｜end▁of▁sentence｜>", "<｜User｜>", "</User|>", "<|User|>", "<|end_of_sentence|>", "</｜"],
        },
        "DeepSeek-R1-Distill-Qwen-14B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "stop_strings": ["<｜end▁of▁sentence｜>", "<｜User｜>", "</User|>", "<|User|>", "<|end_of_sentence|>", "</｜"],
        },
        "DeepSeek-R1-Distill-Qwen-32B": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
            "system_prompt": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "stop_strings": ["<｜end▁of▁sentence｜>", "<｜User｜>", "</User|>", "<|User|>", "<|end_of_sentence|>", "</｜"],
        },
    },
}


compression_configs = {
    "DeepSeek-R1-Distill-Llama-8B": {
        "sym": True,
        "group_size": 128,
        "ratio": 0.8,
    },
    "DeepSeek-R1-Distill-Qwen-7B": {"sym": True, "group_size": 128, "ratio": 1.0},
    "DeepSeek-R1-Distill-Qwen-14B": {"sym": True, "group_size": 128, "ratio": 1.0},
    "DeepSeek-R1-Distill-Qwen-1.5B": {"sym": True, "group_size": 128, "ratio": 1.0},
    "DeepSeek-R1-Distill-Qwen-32B": {"sym": True, "group_size": 128, "ratio": 1.0},
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


def get_llm_selection_widget(
    languages=list(SUPPORTED_LLM_MODELS), models=SUPPORTED_LLM_MODELS[default_language], show_preconverted_checkbox=False, device=None, default_model_idx=-2
):
    import ipywidgets as widgets

    lang_dropdown = widgets.Dropdown(options=languages or [])

    # Define dependent drop down

    model_dropdown = widgets.Dropdown(options=models, value=models[list(models)[default_model_idx]])

    def dropdown_handler(change):
        global default_language
        default_language = change.new
        # If statement checking on dropdown value and changing options of the dependent dropdown accordingly
        model_dropdown.options = SUPPORTED_LLM_MODELS[change.new]
        model_dropdown.value = SUPPORTED_LLM_MODELS[change.new][list(SUPPORTED_LLM_MODELS[change.new])[default_model_idx]]

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


def convert_and_compress_model(model_id, model_config, precision, use_preconverted=False):
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
