import openvino as ov
import openvino_genai as ov_genai
import gradio as gr
import numpy as np
import re
from uuid import uuid4
from threading import Thread
from genai_helper import ChunkStreamer

# ===================== CONFIG =====================

max_new_tokens = 2048
core = ov.Core()

# ===================== EXAMPLES =====================

chinese_examples = [
    ["你好!"],
    ["你是谁?"],
    ["请介绍一下上海"],
    ["请介绍一下英特尔公司"],
    ["晚上睡不着怎么办？"],
    ["给我讲一个年轻人奋斗创业最终取得成功的故事。"],
    ["给这个故事起一个标题。"],
]

english_examples = [
    ["Hello there! How are you doing?"],
    ["What is OpenVINO?"],
    ["Who are you?"],
    ["Can you explain to me briefly what is Python programming language?"],
    ["Explain the plot of Cinderella in a sentence."],
    ["What are some common mistakes to avoid when writing code?"],
    ["Write a 100-word blog post on “Benefits of Artificial Intelligence and OpenVINO“"],
]

japanese_examples = [
    ["こんにちは！調子はどうですか?"],
    ["OpenVINOとは何ですか?"],
    ["あなたは誰ですか?"],
    ["Pythonプログラミング言語とは何か簡単に説明してもらえますか?"],
    ["シンデレラのあらすじを一文で説明してください。"],
    ["コードを書くときに避けるべきよくある間違いは何ですか?"],
    ["人工知能と「OpenVINOの利点」について100語程度のブログ記事を書いてください。"],
]

# ===================== SYSTEM PROMPTS =====================

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.\
"""

DEFAULT_SYSTEM_PROMPT_CHINESE = """\
你是一个乐于助人、尊重他人以及诚实可靠的助手。在安全的情况下，始终尽可能有帮助地回答。
如果一个问题没有任何意义或与事实不符，请解释原因，而不是回答错误的问题。
如果您不知道问题的答案，请不要分享虚假信息。另外，答案请使用中文。\
"""

DEFAULT_SYSTEM_PROMPT_JAPANESE = """\
あなたは親切で、礼儀正しく、誠実なアシスタントです。
質問が意味をなさない場合、または事実に一貫性がない場合は、その理由を説明してください。
質問の答えがわからない場合は、誤った情報を共有しないでください。\
"""

def get_system_prompt(model_language, override=None):
    if override:
        return override
    if model_language == "Chinese":
        return DEFAULT_SYSTEM_PROMPT_CHINESE
    if model_language == "Japanese":
        return DEFAULT_SYSTEM_PROMPT_JAPANESE
    return DEFAULT_SYSTEM_PROMPT

# ===================== DEMO =====================

def make_demo(
    pipe,
    model_configuration,
    model_id,
    model_language,
    disable_advanced=False,
):

    is_vlm = isinstance(pipe, ov_genai.VLMPipeline)

    start_message = get_system_prompt(
        model_language,
        model_configuration.get("start_message"),
    )

    if "genai_chat_template" in model_configuration:
        pipe.get_tokenizer().set_chat_template(
            model_configuration["genai_chat_template"]
        )

    def text_processor(buf, txt):
        txt = re.sub(r"^<think>", "<em><small>I am thinking...", txt)
        txt = re.sub("</think>", "I think I know the answer</small></em>", txt)
        return buf + txt

    def bot(message, history, temperature, top_p, top_k, repetition_penalty, image=None):
        streamer = ChunkStreamer(pipe.get_tokenizer())

        if not disable_advanced:
            config = pipe.get_generation_config()
            config.temperature = temperature
            config.top_p = top_p
            config.top_k = top_k
            config.do_sample = temperature > 0.0
            config.repetition_penalty = repetition_penalty
            config.max_new_tokens = max_new_tokens
        else:
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = max_new_tokens

        history = history or []
        if not history:
            pipe.start_chat(system_message=start_message)

        history.append([message, ""])

        def generate():
            streamer.reset()
            if is_vlm and image is not None:
                pipe.generate(
                    message,
                    image=ov.Tensor(np.array(image)),
                    generation_config=config,
                    streamer=streamer,
                )
            else:
                pipe.generate(
                    message,
                    generation_config=config,
                    streamer=streamer,
                )
            streamer.end()

        Thread(target=generate).start()

        buf = ""
        for chunk in streamer:
            buf = text_processor(buf, chunk)
            history[-1][1] = buf
            yield "", history, streamer, image, image

    def stop(streamer):
        if streamer:
            streamer.end()
        return None

    def clear(streamer):
        if streamer:
            streamer.end()
        pipe.finish_chat()
        return None, None, None, None, None

    examples = (
        chinese_examples if model_language == "Chinese"
        else japanese_examples if model_language == "Japanese"
        else english_examples
    )

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        streamer = gr.State(None)
        image_state = gr.State(None)

        gr.Markdown(f"# OpenVINO {model_id} Chatbot")

        chatbot = gr.Chatbot(height=500)

        with gr.Row():
            msg = gr.Textbox(placeholder="Type a message...", scale=5)
            attach = gr.UploadButton(
                "📎 Image",
                file_types=["image"],
                file_count="single",
                interactive=is_vlm,
                scale=1,
            )
            submit = gr.Button("Submit", scale=1)

        with gr.Row():
            stop_btn = gr.Button("Stop")
            clear_btn = gr.Button("Clear")

        image_preview = gr.Image(height=120, interactive=False)

        with gr.Accordion("Advanced Options", open=False, visible=not disable_advanced):
            temperature = gr.Slider(0.0, 1.0, 0.0, step=0.1, label="Temperature")
            top_p = gr.Slider(0.01, 1.0, 1.0, step=0.01, label="Top-p")
            top_k = gr.Slider(0, 200, 1, step=1, label="Top-k")
            repetition_penalty = gr.Slider(1.0, 2.0, 1.1, step=0.1, label="Repetition Penalty")

        gr.Examples(examples, inputs=msg)

        def handle_attach(file):
            if not file:
                return None, None
            from PIL import Image
            img = Image.open(file["name"]) if isinstance(file, dict) else Image.open(file)
            return img, img

        attach.upload(handle_attach, attach, [image_state, image_preview], queue=False)

        msg.submit(
            bot,
            [msg, chatbot, temperature, top_p, top_k, repetition_penalty, image_state],
            [msg, chatbot, streamer, image_state, image_preview],
            queue=True,
        )

        submit.click(
            bot,
            [msg, chatbot, temperature, top_p, top_k, repetition_penalty, image_state],
            [msg, chatbot, streamer, image_state, image_preview],
            queue=True,
        )

        stop_btn.click(stop, streamer, streamer, queue=False)
        clear_btn.click(clear, streamer, [chatbot, streamer, image_state, image_preview, msg], queue=False)

    return demo
