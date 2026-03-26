import re
import time

import openvino as ov
import openvino_genai as ov_genai
from threading import Thread
from pathlib import Path
from genai_helper import ChunkStreamer, load_video_frames
import numpy as np
from PIL import Image


def discover_available_models(model_id, model_dir, model_configuration, supported_vlm_models):
    """Discover all already-converted VLM models on disk.

    Returns a dict of {display_key: {"model_dir": Path, "config": dict}}.
    The currently loaded model is always included.
    """
    model_dir = Path(model_dir)
    current_precision = model_dir.name.replace("_compressed_weights", "")
    available = {
        f"{model_id} ({current_precision})": {"model_dir": model_dir, "config": model_configuration},
    }

    dir_to_info = {}
    for lang, models in supported_vlm_models.items():
        for name, cfg in models.items():
            dirname = re.sub(r'[<>:"/\\|?*]', "_", name)
            if dirname not in dir_to_info:
                dir_to_info[dirname] = (name, cfg)

    for parent in Path(".").iterdir():
        if parent.is_dir() and parent.name in dir_to_info:
            name, cfg = dir_to_info[parent.name]
            for sub in sorted(parent.iterdir()):
                if sub.is_dir() and (sub / "openvino_language_model.xml").exists():
                    precision = sub.name.replace("_compressed_weights", "")
                    key = f"{name} ({precision})"
                    if key not in available:
                        available[key] = {"model_dir": sub, "config": cfg}

    return available


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

    def _strip_think_tags(text):
        """Remove <think> and </think> tags from text, keeping content."""
        return text.replace("<think>", "").replace("</think>", "")

    def bot(message, history, temperature, top_p, top_k, repetition_penalty, enable_thinking):
        current_pipe = state["pipe"]
        current_config = state["config"]

        text = (message.get("text") or "").strip()
        files = message.get("files") or []

        model_supports_thinking = current_config.get("supports_thinking", False)
        show_thinking = enable_thinking and model_supports_thinking
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
            if show_thinking:
                # Qwen3-VL-Thinking recommended params
                config.temperature = 1.0
                config.top_p = 0.95
                config.top_k = 20
                config.do_sample = True
            else:
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

        def generate_and_signal_complete():
            try:
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
            except Exception as e:
                print(f"Generation error: {e}")
            finally:
                streamer.end()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        partial_text = ""

        if not show_thinking:
            # --- Non-thinking path (or checkbox off) ---
            # Strip <think>/<​/think> tags, show content as plain text.
            for new_text in streamer:
                partial_text = text_processor(partial_text, new_text)
                history[-1]["content"] = _strip_think_tags(partial_text)
                yield gr.MultimodalTextbox(value=None), history, streamer
        else:
            # --- Thinking path: collapsible block via gr.ChatMessage ---
            # The Thinking model's chat template injects <think>\n at the
            # end of the prompt, so the model output starts with reasoning
            # immediately (no <think> tag in the streamed text).  We only
            # need to detect </think> as the boundary.  If the model
            # happens to emit <think> explicitly, we handle that too.
            thinking_started = time.time()
            thinking_msg = gr.ChatMessage(
                role="assistant",
                content="",
                metadata={"title": "\U0001f914 Thinking", "status": "pending"},
            )
            history[-1] = thinking_msg
            think_done = False
            answer_start_pos = 0  # cached position where answer text begins after </think>

            for new_text in streamer:
                partial_text = text_processor(partial_text, new_text)

                # Strip leading <think> if model echoes it
                thinking_text = partial_text
                if thinking_text.startswith("<think>"):
                    thinking_text = thinking_text[len("<think>") :]

                if not think_done:
                    if "</think>" in thinking_text:
                        # Thinking finished — split reasoning from answer
                        think_done = True
                        parts = thinking_text.split("</think>", 1)
                        duration = time.time() - thinking_started if thinking_started else 0
                        thinking_content = parts[0].strip()
                        thinking_msg.metadata["status"] = "done"
                        thinking_msg.metadata["duration"] = round(duration, 1)
                        answer_text = parts[1].strip() if len(parts) > 1 else ""
                        answer_start_pos = len(parts[0]) + len("</think>")
                        if thinking_content and answer_text:
                            # Both thinking and answer present
                            thinking_msg.content = thinking_content
                            history.append({"role": "assistant", "content": answer_text})
                        elif thinking_content and not answer_text:
                            # Thinking present, no answer yet — keep both;
                            # answer will be streamed into history[-1] below
                            thinking_msg.content = thinking_content
                            history.append({"role": "assistant", "content": ""})
                        elif not thinking_content and answer_text:
                            # Empty thinking (model did <think></think>) — show answer only
                            history[-1] = {"role": "assistant", "content": answer_text}
                            thinking_msg = None
                        else:
                            # Both empty — just add empty answer slot
                            history[-1] = {"role": "assistant", "content": ""}
                            thinking_msg = None
                        yield gr.MultimodalTextbox(value=None), history, streamer
                        continue

                    # Still accumulating thinking content
                    thinking_msg.content = thinking_text
                    yield gr.MultimodalTextbox(value=None), history, streamer
                else:
                    # After </think> — use cached position to extract answer
                    history[-1]["content"] = thinking_text[answer_start_pos:].strip()
                    yield gr.MultimodalTextbox(value=None), history, streamer

            # End of stream: handle case where </think> never appeared
            if not think_done and thinking_msg is not None:
                content = thinking_msg.content.strip() if thinking_msg.content else ""
                if content:
                    # Model never closed </think> — the content IS the answer
                    history[-1] = {"role": "assistant", "content": content}
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
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()

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
        new_pipe = ov_genai.VLMPipeline(str(info["model_dir"]), device)
        _init_pipe(new_pipe, new_config)

        state["pipe"] = new_pipe
        state["config"] = new_config
        state["model_id"] = selected_model

        new_title = f"""<h1><center>OpenVINO {selected_model} Chatbot</center></h1>"""
        thinks = new_config.get("supports_thinking", False)
        return new_title, [], None, gr.Checkbox(value=False, visible=thinks)

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
        enable_thinking = gr.Checkbox(
            label="Enable Thinking",
            value=False,
            visible=model_configuration.get("supports_thinking", False),
            interactive=not disable_advanced,
        )
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

        if show_model_selector:
            # Chain: lock dropdown → run bot → unlock dropdown
            disable_event = msg.submit(lambda: gr.Dropdown(interactive=False), outputs=[model_selector], queue=False)
            submit_event = disable_event.then(
                fn=bot,
                inputs=[msg, chatbot, temperature, top_p, top_k, repetition_penalty, enable_thinking],
                outputs=[msg, chatbot, streamer],
            )
            submit_event.then(lambda: gr.Dropdown(interactive=True), outputs=[model_selector])
        else:
            msg.submit(
                fn=bot,
                inputs=[msg, chatbot, temperature, top_p, top_k, repetition_penalty, enable_thinking],
                outputs=[msg, chatbot, streamer],
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
                outputs=[title_md, chatbot, streamer, enable_thinking],
            )

        return demo
