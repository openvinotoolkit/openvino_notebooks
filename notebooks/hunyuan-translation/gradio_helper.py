import queue
import openvino_genai as ov_genai
from threading import Thread

max_new_tokens = 2048

LANGUAGE_MAP = {
    "Chinese": "中文",
    "English": "English",
    "French": "French",
    "German": "German",
    "Spanish": "Spanish",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "Russian": "Russian",
    "Arabic": "Arabic",
    "Portuguese": "Portuguese",
    "Italian": "Italian",
    "Thai": "Thai",
    "Vietnamese": "Vietnamese",
    "Turkish": "Turkish",
    "Indonesian": "Indonesian",
    "Malay": "Malay",
    "Hindi": "Hindi",
    "Polish": "Polish",
    "Dutch": "Dutch",
    "Czech": "Czech",
    "Filipino": "Filipino",
    "Bengali": "Bengali",
    "Tamil": "Tamil",
    "Ukrainian": "Ukrainian",
    "Persian": "Persian",
    "Hebrew": "Hebrew",
    "Urdu": "Urdu",
    "Khmer": "Khmer",
    "Burmese": "Burmese",
    "Gujarati": "Gujarati",
    "Telugu": "Telugu",
    "Marathi": "Marathi",
    "Traditional Chinese": "繁体中文",
}

CHINESE_LANGUAGES = {"Chinese", "Traditional Chinese"}

LANGUAGE_LIST = list(LANGUAGE_MAP.keys())

EXAMPLES = [
    ["OpenVINO is a toolkit for optimizing and deploying deep learning models.", "English", "Chinese"],
    ["人工智能正在深刻改变我们的生活方式和工作方式。", "Chinese", "English"],
    ["The quick brown fox jumps over the lazy dog.", "English", "Japanese"],
    ["机器翻译技术已经取得了显著的进步。", "Chinese", "French"],
    ["Climate change is one of the biggest challenges facing humanity today.", "English", "German"],
    ["今天天气真好，适合出去散步。", "Chinese", "Korean"],
]


def build_translation_prompt(source_text, source_lang, target_lang):
    """Build the translation prompt following HY-MT1.5 format."""
    target_name = LANGUAGE_MAP.get(target_lang, target_lang)

    if source_lang in CHINESE_LANGUAGES or target_lang in CHINESE_LANGUAGES:
        prompt = f"将以下文本翻译为{target_name}，注意只需要输出翻译后的结果，不要额外解释：\n\n{source_text}"
    else:
        prompt = f"Translate the following segment into {target_name}, without additional explanation.\n\n{source_text}"

    return prompt


def make_demo(pipe, model_name):
    import gradio as gr

    _STOP = object()

    def translate(source_text, source_lang, target_lang, temperature, top_p, top_k, repetition_penalty):
        if not source_text.strip():
            yield ""
            return

        prompt = build_translation_prompt(source_text, source_lang, target_lang)

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_new_tokens
        config.temperature = temperature
        config.top_p = top_p
        config.top_k = int(top_k)
        config.do_sample = temperature > 0.0
        config.repetition_penalty = repetition_penalty

        text_queue = queue.Queue()

        def callback(subword):
            text_queue.put(subword)
            return False

        def generate_in_thread():
            pipe.generate(prompt, config, callback)
            text_queue.put(_STOP)

        t1 = Thread(target=generate_in_thread)
        t1.start()

        partial_text = ""
        while True:
            item = text_queue.get()
            if item is _STOP:
                break
            partial_text += item
            yield partial_text

        t1.join(timeout=30)

    def swap_languages(source_lang, target_lang, source_text, target_text):
        return target_lang, source_lang, target_text, source_text

    with gr.Blocks(
        title=f"Hunyuan Translation ({model_name})",
    ) as demo:
        gr.Markdown(
            f"""<h1><center>OpenVINO Hunyuan Translation ({model_name})</center></h1>
<p><center>Powered by <a href="https://huggingface.co/tencent/HY-MT1.5-1.8B">HY-MT1.5</a> and <a href="https://github.com/openvinotoolkit/openvino.genai">OpenVINO GenAI</a> | Supports 33+ languages</center></p>"""
        )

        with gr.Row():
            source_lang = gr.Dropdown(
                choices=LANGUAGE_LIST,
                value="English",
                label="Source Language",
                scale=2,
            )
            swap_btn = gr.Button("⇄ Swap", scale=1, min_width=80)
            target_lang = gr.Dropdown(
                choices=LANGUAGE_LIST,
                value="Chinese",
                label="Target Language",
                scale=2,
            )

        with gr.Row():
            with gr.Column():
                source_text = gr.Textbox(
                    label="Source Text",
                    placeholder="Enter text to translate...",
                    lines=8,
                    max_lines=20,
                )
            with gr.Column():
                target_text = gr.Textbox(
                    label="Translation",
                    lines=8,
                    max_lines=20,
                    interactive=False,
                )

        with gr.Row():
            translate_btn = gr.Button("Translate", variant="primary", scale=2)
            clear_btn = gr.Button("Clear", scale=1)

        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                temperature = gr.Slider(
                    label="Temperature",
                    value=0.7,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    info="Higher values produce more diverse outputs",
                )
                top_p = gr.Slider(
                    label="Top-p",
                    value=0.6,
                    minimum=0.01,
                    maximum=1.0,
                    step=0.01,
                    info="Nucleus sampling threshold",
                )
                top_k = gr.Slider(
                    label="Top-k",
                    value=20,
                    minimum=0,
                    maximum=200,
                    step=1,
                    info="Number of top tokens to consider",
                )
                repetition_penalty = gr.Slider(
                    label="Repetition Penalty",
                    value=1.05,
                    minimum=1.0,
                    maximum=2.0,
                    step=0.05,
                    info="Penalize repetition — 1.0 to disable",
                )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[source_text, source_lang, target_lang],
            label="Click on any example and press 'Translate'",
        )

        translate_btn.click(
            fn=translate,
            inputs=[source_text, source_lang, target_lang, temperature, top_p, top_k, repetition_penalty],
            outputs=[target_text],
            queue=True,
        )

        source_text.submit(
            fn=translate,
            inputs=[source_text, source_lang, target_lang, temperature, top_p, top_k, repetition_penalty],
            outputs=[target_text],
            queue=True,
        )

        swap_btn.click(
            fn=swap_languages,
            inputs=[source_lang, target_lang, source_text, target_text],
            outputs=[source_lang, target_lang, source_text, target_text],
        )

        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[source_text, target_text],
        )

    return demo
