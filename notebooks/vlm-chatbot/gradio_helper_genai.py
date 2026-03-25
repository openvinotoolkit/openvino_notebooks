import shutil

import openvino as ov
import openvino_genai as ov_genai
from threading import Event, Thread
from pathlib import Path
from genai_helper import ChunkStreamer, load_video_frames
import numpy as np
from PIL import Image


def _ensure_video_processor_config(model_dir):
    """Workaround: copy preprocessor_config.json → video_preprocessor_config.json
    if missing, so GenAI picks up the correct patch_size for video.
    Only needed for Qwen-VL family models (qwen2_vl, qwen2_5_vl, qwen3_vl)."""
    import json

    model_dir = Path(model_dir)
    video_cfg = model_dir / "video_preprocessor_config.json"
    if video_cfg.exists():
        return
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return
    with open(config_path) as f:
        model_type = json.load(f).get("model_type", "")
    if "qwen" not in model_type:
        return
    image_cfg = model_dir / "preprocessor_config.json"
    if image_cfg.exists():
        shutil.copy2(image_cfg, video_cfg)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

max_new_tokens = 2048

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


def get_system_prompt(model_language, system_prompt=None):
    if system_prompt is not None:
        return system_prompt
    return (
        DEFAULT_SYSTEM_PROMPT_CHINESE
        if (model_language == "Chinese")
        else (DEFAULT_SYSTEM_PROMPT_JAPANESE if (model_language == "Japanese") else DEFAULT_SYSTEM_PROMPT)
    )


def make_demo(
    pipe, model_configuration, model_id, model_language, disable_advanced=False, sample_image=None, sample_video=None, available_models=None, device="CPU"
):
    import gradio as gr
    import gc

    max_new_tokens = 2048

    # Mutable state — allows model switching at runtime
    state = {
        "pipe": pipe,
        "config": model_configuration,
        "model_id": model_id,
    }

    def _init_pipe(p, cfg):
        if "genai_chat_template" in cfg:
            p.get_tokenizer().set_chat_template(cfg["genai_chat_template"])

    _init_pipe(pipe, model_configuration)

    def default_partial_text_processor(partial_text, new_text):
        partial_text += new_text
        return partial_text

    # Aggregate capabilities across all available models for UI setup
    any_video = model_configuration.get("supports_video", False)
    if available_models:
        any_video = any_video or any(m["config"].get("supports_video", False) for m in available_models.values())
    file_types = ["image", ".mp4"] if any_video else ["image"]
    show_model_selector = available_models and len(available_models) > 1

    def bot(message, history, temperature, top_p, top_k, repetition_penalty):
        current_pipe = state["pipe"]
        current_config = state["config"]

        text = (message.get("text") or "").strip()
        files = message.get("files") or []

        # Build prompt — keep text as-is, thinking is controlled via GenerationConfig
        prompt_text = text

        cur_supports_video = current_config.get("supports_video", False)
        images, videos = [], []
        for f in files:
            fpath = f if isinstance(f, str) else f.get("path", "") if isinstance(f, dict) else ""
            if fpath.lower().endswith(VIDEO_EXTENSIONS):
                if cur_supports_video:
                    videos.append(fpath)
            elif fpath:
                images.append(fpath)

        text_processor = current_config.get("partial_text_processor", default_partial_text_processor)

        streamer = ChunkStreamer(current_pipe.get_tokenizer())
        if not disable_advanced:
            config = current_pipe.get_generation_config()
            config.temperature = temperature
            config.top_p = top_p
            config.top_k = top_k
            config.do_sample = temperature > 0.0
            config.max_new_tokens = max_new_tokens
            config.repetition_penalty = repetition_penalty
        else:
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = max_new_tokens

        history = history or []
        if not history:
            start_msg = get_system_prompt(model_language, current_config.get("start_message"))
            current_pipe.start_chat(system_message=start_msg)

        for fpath in images:
            history.append({"role": "user", "content": gr.Image(fpath)})
        for fpath in videos:
            history.append({"role": "user", "content": gr.Video(fpath)})
        if text:
            history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": ""})

        stream_complete = Event()

        def generate_and_signal_complete():
            streamer.reset()
            image_tensors = [ov.Tensor(np.array(Image.open(p).convert("RGB"))) for p in images]
            video_tensors = [load_video_frames(v) for v in videos]

            if video_tensors and image_tensors:
                current_pipe.generate(prompt_text, images=image_tensors, videos=video_tensors, generation_config=config, streamer=streamer)
            elif video_tensors:
                current_pipe.generate(prompt_text, videos=video_tensors, generation_config=config, streamer=streamer)
            elif len(image_tensors) == 1:
                current_pipe.generate(prompt_text, image=image_tensors[0], generation_config=config, streamer=streamer)
            elif image_tensors:
                current_pipe.generate(prompt_text, images=image_tensors, generation_config=config, streamer=streamer)
            else:
                current_pipe.generate(prompt_text, generation_config=config, streamer=streamer)
            stream_complete.set()
            streamer.end()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        partial_text = ""
        for new_text in streamer:
            partial_text = text_processor(partial_text, new_text)
            history[-1]["content"] = partial_text
            yield gr.MultimodalTextbox(value=None), history, streamer

    def stop_chat(streamer):
        if streamer is not None:
            streamer.end()
        return None

    def stop_chat_and_clear_history(streamer):
        if streamer is not None:
            streamer.end()
        state["pipe"].finish_chat()
        return [], None

    def switch_model(selected_model, current_streamer):
        if not available_models or selected_model not in available_models:
            return gr.skip(), gr.skip(), gr.skip()

        info = available_models[selected_model]
        new_config = info["config"]

        if current_streamer is not None:
            current_streamer.end()

        try:
            state["pipe"].finish_chat()
        except Exception:
            pass
        del state["pipe"]
        gc.collect()

        gr.Info(f"Loading {selected_model}…")
        _ensure_video_processor_config(info["model_dir"])
        new_pipe = ov_genai.VLMPipeline(str(info["model_dir"]), device)
        _init_pipe(new_pipe, new_config)

        state["pipe"] = new_pipe
        state["config"] = new_config
        state["model_id"] = selected_model

        new_title = f"""<h1><center>OpenVINO {selected_model} Chatbot</center></h1>"""
        return new_title, [], None

    text_examples = chinese_examples if (model_language == "Chinese") else japanese_examples if (model_language == "Japanese") else english_examples
    examples = []

    if any_video and sample_video and Path(sample_video).exists():
        vid_prompt = (
            "描述视频中发生的事情。"
            if model_language == "Chinese"
            else "このビデオで何が起きていますか？" if model_language == "Japanese" else "Describe what is happening in this video."
        )
        examples.append([{"text": vid_prompt, "files": [str(sample_video)]}])

    if sample_image and Path(sample_image).exists():
        img_prompt = (
            "这张图片里有什么？" if model_language == "Chinese" else "この画像には何がありますか？" if model_language == "Japanese" else "What is on the image?"
        )
        examples.append([{"text": img_prompt, "files": [str(sample_image)]}])

    for ex in text_examples:
        examples.append([{"text": ex[0], "files": []}])

    with gr.Blocks() as demo:
        streamer = gr.State(None)
        title_md = gr.Markdown(f"""<h1><center>OpenVINO {model_id} Chatbot</center></h1>""")

        if show_model_selector:
            current_key = next(
                (k for k in available_models if model_id in k),
                list(available_models.keys())[0],
            )
            model_selector = gr.Dropdown(
                choices=list(available_models.keys()),
                value=current_key,
                label="Model",
            )

        chatbot = gr.Chatbot(
            height=500,
        )
        msg = gr.MultimodalTextbox(
            file_types=file_types,
            file_count="multiple",
            placeholder="Ask a question — attach images or video if needed",
            show_label=False,
        )
        with gr.Row():
            stop = gr.Button("Stop")
            clear = gr.Button("Clear")
        with gr.Row(visible=not disable_advanced):
            with gr.Accordion("Advanced Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.0,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=1.0,
                                minimum=0.01,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sample from the smallest possible set of tokens whose cumulative probability "
                                    "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k",
                                value=1,
                                minimum=0.0,
                                maximum=200,
                                step=1,
                                interactive=True,
                                info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.1,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )
        gr.Examples(
            examples,
            inputs=[msg],
            label="Click on any example and press the 'Submit' button",
        )

        msg.submit(
            fn=bot,
            inputs=[msg, chatbot, temperature, top_p, top_k, repetition_penalty],
            outputs=[msg, chatbot, streamer],
            queue=True,
        )
        stop.click(fn=stop_chat, inputs=streamer, outputs=[streamer], queue=False)
        clear.click(
            fn=stop_chat_and_clear_history,
            inputs=streamer,
            outputs=[chatbot, streamer],
            queue=False,
        )
        if show_model_selector:
            model_selector.change(
                fn=switch_model,
                inputs=[model_selector, streamer],
                outputs=[title_md, chatbot, streamer],
            )

        return demo
