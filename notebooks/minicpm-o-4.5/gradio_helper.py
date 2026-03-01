"""
Gradio helper for MiniCPM-o 4.5 OpenVINO notebook demo.

Provides a rich multimodal chatbot UI supporting:
  - Text, image, and audio inputs
  - Streaming & non-streaming text generation
  - Stop button for interrupting generation
  - Thinking-mode toggle (<think>...</think> parsing)
  - Few-shot learning tab
  - Sampling parameter panel
  - Regenerate / Clear History

Inspired by: https://github.com/OpenSQZ/MiniCPM-V-CookBook (gradio client)
Adapted for direct OpenVINO model calls (no client/server).
"""

from copy import deepcopy
import re
from PIL import Image
import librosa
import numpy as np
import gradio as gr

# ──────────────────── Constants ──────────────────────────────────────────

MODEL_NAME = "MiniCPM-o 4.5 (OpenVINO)"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".wma", ".ogg")
MAX_NEW_TOKENS = 2048


# ──────────────────── Utility functions ─────────────────────────────────

def parse_thinking_response(text: str):
    """Parse <think>...</think> blocks from model output."""
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, text, re.DOTALL)
    thinking = "\n\n".join(m.strip() for m in matches) if matches else ""
    answer = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return thinking, answer


def format_response(thinking: str, answer: str) -> str:
    """Format response with optional thinking section (HTML)."""
    if thinking:
        return (
            '<div class="response-container">'
            '<details class="thinking-section" open>'
            '<summary class="thinking-header">💭 Thinking</summary>'
            f'<div class="thinking-content">{thinking}</div>'
            '</details>'
            f'<div class="answer-section">{answer}</div>'
            '</div>'
        )
    return answer


def classify_file(path: str):
    """Return 'image' or 'audio' or None based on extension."""
    if path.lower().endswith(IMAGE_EXTENSIONS):
        return "image"
    if path.lower().endswith(AUDIO_EXTENSIONS):
        return "audio"
    return None


# ──────────────────── Message conversion ────────────────────────────────

def history_to_messages(history: list) -> list:
    """Convert Gradio chat history (type='messages') to model message format."""
    messages = []
    cur = {}
    for item in history:
        role = item.get("role")
        if role == "assistant":
            if cur:
                messages.append(deepcopy(cur))
                cur = {}
            messages.append({"role": "assistant", "content": item["content"]})
            continue
        if "role" not in cur:
            cur = {"role": "user", "content": []}
        metadata = item.get("metadata", {})
        title = metadata.get("title")
        if title == "image":
            cur["content"].append(Image.open(item["content"][0]).convert("RGB"))
        elif title == "audio":
            audio_data, _ = librosa.load(item["content"][0], sr=16000, mono=True)
            cur["content"].append(audio_data)
        elif title is None:
            cur["content"].append(item["content"])
    if cur:
        messages.append(cur)
    return messages


# ──────────────────── Input validation ──────────────────────────────────

def check_messages(history, message, audio):
    """Validate & append user messages; return updated history."""
    has_text = message.get("text", "").strip() if isinstance(message, dict) else False
    has_files = len(message.get("files", [])) > 0 if isinstance(message, dict) else False
    has_audio = audio is not None

    if not (has_text or has_files or has_audio):
        raise gr.Error("Message is empty — please enter text, upload a file, or record audio.")

    images, audios = [], []
    for fpath in message.get("files", []):
        ftype = classify_file(fpath)
        if ftype == "image":
            images.append(fpath)
        elif ftype == "audio":
            dur = librosa.get_duration(filename=fpath)
            if dur > 60:
                raise gr.Error("Audio too long (>60 s). Please use a shorter clip.")
            audios.append(fpath)
        else:
            raise gr.Error(f"Unsupported file type: {fpath.split('/')[-1]}")

    if len(audios) > 1:
        raise gr.Error("Only one audio file per turn is supported.")
    if audio is not None:
        if audios:
            raise gr.Error("Upload OR record audio — not both.")
        audios.append(audio)

    for img in images:
        history.append({"role": "user", "content": (img,), "metadata": {"title": "image"}})
    for aud in audios:
        history.append({"role": "user", "content": (aud,), "metadata": {"title": "audio"}})
    if has_text:
        history.append({"role": "user", "content": message["text"], "metadata": {}})

    return history, gr.MultimodalTextbox(value=None, interactive=False), None


# ──────────────────── Few-shot helpers ──────────────────────────────────

def add_fewshot_example(image, user_msg, assistant_msg, history):
    """Add a user+assistant example pair to the few-shot context."""
    if not user_msg and image is None:
        raise gr.Error("Provide at least image or text for the example.")
    if image is not None:
        history.append({"role": "user", "content": (image,), "metadata": {"title": "image"}})
    if user_msg:
        history.append({"role": "user", "content": user_msg, "metadata": {}})
    if assistant_msg:
        history.append({"role": "assistant", "content": assistant_msg})
    return None, "", "", history


# ──────────────────── CSS ───────────────────────────────────────────────

CSS = """
.response-container { margin: 4px 0; }
.thinking-section {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%);
    border: 1px solid #c7d2fe; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
}
.thinking-header {
    font-weight: 600; color: #4338ca; font-size: 13px;
    cursor: pointer; list-style: none;
}
.thinking-header::-webkit-details-marker { display: none; }
.thinking-content {
    color: #6366f1; font-size: 13px; line-height: 1.55;
    font-style: italic; padding: 8px 12px; margin-top: 6px;
    background: rgba(255,255,255,0.5); border-radius: 6px;
    border-left: 3px solid #818cf8; white-space: pre-wrap;
}
.answer-section {
    font-size: 14px; line-height: 1.6; color: #1e293b; white-space: pre-wrap;
}
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
.header-banner {
    text-align: center; padding: 18px 0 8px;
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(34,211,238,0.04));
    border-radius: 14px; margin-bottom: 6px;
    border: 1px solid rgba(99,102,241,0.15);
}
.header-banner h1 {
    font-size: 1.75rem; font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 2px;
}
.header-banner p { color: #64748b; font-size: 0.85rem; margin: 0; }
"""


# ──────────────────── make_demo ─────────────────────────────────────────

def make_demo(ov_model):
    """Create and return the Gradio Blocks demo."""

    stop_flag = {"value": False}

    def bot(
        history, top_p, top_k, temperature, repetition_penalty,
        max_tokens, thinking_mode, streaming_mode, regenerate=False,
    ):
        if history and regenerate:
            while history and history[-1].get("role") == "assistant":
                history.pop()
        if not history:
            return history

        stop_flag["value"] = False
        msgs = history_to_messages(history)
        has_audio = any(
            isinstance(c, np.ndarray)
            for m in msgs if m["role"] == "user"
            for c in (m["content"] if isinstance(m["content"], list) else [])
        )

        gen_cfg = {
            "top_p": top_p, "top_k": top_k,
            "temperature": temperature, "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
        }
        if thinking_mode:
            gen_cfg["enable_thinking"] = True

        ov_model.llm._ov_language.reset_state()
        ov_model.llm._past_length = 0
        history.append({"role": "assistant", "content": ""})

        if streaming_mode:
            res = ov_model.chat(
                msgs=msgs, max_new_tokens=max_tokens, stream=True,
                use_tts_template=has_audio, **gen_cfg,
            )
            raw = ""
            for chunk in res:
                if stop_flag["value"]:
                    break
                raw += chunk
                if thinking_mode:
                    tk, ans = parse_thinking_response(raw)
                    history[-1]["content"] = format_response(tk, ans)
                else:
                    history[-1]["content"] = raw
                yield history
        else:
            answer = ov_model.chat(
                msgs=msgs, max_new_tokens=max_tokens, stream=False,
                use_tts_template=has_audio, **gen_cfg,
            )
            if thinking_mode:
                tk, ans = parse_thinking_response(answer)
                history[-1]["content"] = format_response(tk, ans)
            else:
                history[-1]["content"] = answer
            yield history

    def fewshot_generate(image, user_msg, history,
                         top_p, top_k, temperature, rep_pen,
                         max_tokens, thinking_mode, streaming_mode):
        if image is not None:
            history.append({"role": "user", "content": (image,), "metadata": {"title": "image"}})
        if user_msg:
            history.append({"role": "user", "content": user_msg, "metadata": {}})
        if not history:
            yield None, "", history
            return
        for h in bot(history, top_p, top_k, temperature, rep_pen,
                     max_tokens, thinking_mode, streaming_mode):
            yield image, user_msg, h

    def on_stop():
        stop_flag["value"] = True

    # ──────────────────── UI ────────────────────────────────────────────

    with gr.Blocks(
        title=f"Chat with {MODEL_NAME}",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.cyan,
            neutral_hue=gr.themes.colors.slate,
        ),
        css=CSS,
    ) as demo:
        gr.HTML(
            '<div class="header-banner">'
            f"<h1>🪐 {MODEL_NAME}</h1>"
            "<p>Multimodal Chat — Text · Image · Audio | Streaming · Thinking Mode · Few-Shot</p>"
            "</div>"
        )

        with gr.Tab("💬 Chat"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=260):
                    decode_type = gr.Radio(
                        ["Sampling", "Beam Search"], value="Sampling", label="Decode Type",
                    )
                    thinking_toggle = gr.Checkbox(
                        value=False, label="🧠 Enable Thinking Mode",
                        info="Model shows its reasoning process",
                    )
                    streaming_toggle = gr.Checkbox(
                        value=True, label="⚡ Enable Streaming",
                        info="Real-time token output",
                    )
                    with gr.Group():
                        gr.Markdown("#### 🎛️ Sampling Parameters")
                        temperature = gr.Slider(0, 1, value=0.7, label="Temperature")
                        top_p = gr.Slider(0, 1, value=0.8, label="Top-p")
                        top_k = gr.Slider(0, 1000, value=100, step=1, label="Top-k")
                        rep_penalty = gr.Slider(0.5, 2.0, value=1.05, step=0.01, label="Repetition Penalty")
                        max_tokens = gr.Slider(64, 4096, value=MAX_NEW_TOKENS, step=64, label="Max New Tokens")
                    with gr.Row():
                        regenerate_btn = gr.Button("🔄 Regenerate", variant="secondary")
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    stop_btn = gr.Button("⏹ Stop Generation", variant="stop", visible=False)

                with gr.Column(scale=3, min_width=500):
                    chatbot = gr.Chatbot(
                        label=f"Chat with {MODEL_NAME}",
                        elem_id="chatbot", bubble_full_width=False,
                        type="messages", height="56vh",
                        show_copy_button=True, sanitize_html=False,
                    )
                    chat_input = gr.MultimodalTextbox(
                        file_count="multiple",
                        placeholder="Enter text or upload image/audio …",
                        show_label=False, file_types=["image", "audio"],
                        interactive=True,
                    )
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"], type="filepath",
                        max_length=30, label="🎤 Record or upload audio",
                    )

            def disable_streaming_on_beam(decode):
                if decode == "Beam Search":
                    return gr.update(value=False, interactive=False,
                                     info="Beam Search does not support streaming")
                return gr.update(value=True, interactive=True, info="Real-time token output")

            decode_type.change(disable_streaming_on_beam, decode_type, streaming_toggle)

            chat_msg = chat_input.submit(
                check_messages, [chatbot, chat_input, audio_input],
                [chatbot, chat_input, audio_input],
            )
            bot_gen = chat_msg.then(
                lambda: gr.update(visible=True), None, stop_btn,
            ).then(
                bot,
                [chatbot, top_p, top_k, temperature, rep_penalty,
                 max_tokens, thinking_toggle, streaming_toggle],
                chatbot,
            ).then(
                lambda: (gr.update(visible=False), gr.MultimodalTextbox(interactive=True)),
                None, [stop_btn, chat_input],
            )
            stop_btn.click(on_stop, None, None, cancels=[bot_gen])

            regenerate_btn.click(
                lambda: gr.update(visible=True), None, stop_btn,
            ).then(
                bot,
                [chatbot, top_p, top_k, temperature, rep_penalty,
                 max_tokens, thinking_toggle, streaming_toggle, gr.State(True)],
                chatbot,
            ).then(lambda: gr.update(visible=False), None, stop_btn)

            clear_btn.click(lambda: ([], None, None), None, [chatbot, chat_input, audio_input])

        with gr.Tab("📚 Few-Shot Learning"):
            gr.Markdown(
                "Add example image+answer pairs, then ask a new question. "
                "The model will learn the pattern and apply it."
            )
            with gr.Row():
                with gr.Column(scale=3, min_width=500):
                    fs_chatbot = gr.Chatbot(
                        label="Few-Shot Conversation", type="messages",
                        height="50vh", bubble_full_width=False,
                        show_copy_button=True, sanitize_html=False,
                    )
                with gr.Column(scale=1, min_width=260):
                    fs_image = gr.Image(type="filepath", sources=["upload"], label="Example Image")
                    fs_user = gr.Textbox(label="User Message")
                    fs_assistant = gr.Textbox(label="Assistant Answer")
                    fs_add_btn = gr.Button("➕ Add Example")
                    fs_gen_btn = gr.Button("🚀 Generate", variant="primary")
                    fs_clear_btn = gr.Button("🗑️ Clear All")

            fs_add_btn.click(
                add_fewshot_example,
                [fs_image, fs_user, fs_assistant, fs_chatbot],
                [fs_image, fs_user, fs_assistant, fs_chatbot],
            )
            fs_gen_btn.click(
                fewshot_generate,
                [fs_image, fs_user, fs_chatbot, top_p, top_k, temperature,
                 rep_penalty, max_tokens, thinking_toggle, streaming_toggle],
                [fs_image, fs_user, fs_chatbot],
            )
            fs_clear_btn.click(lambda: (None, "", "", []), None,
                               [fs_image, fs_user, fs_assistant, fs_chatbot])

        with gr.Tab("📖 How to Use"):
            gr.Markdown(f"""
### {MODEL_NAME} — Multimodal Chat

**Chat Tab**
1. Type a question or upload an image / audio file in the input box.
2. Press **Enter** to send. The model will stream its response in real-time.
3. Toggle **🧠 Thinking Mode** to see the model's reasoning process.
4. Use **⏹ Stop** to interrupt generation at any time.
5. Adjust sampling parameters (Temperature, Top-p, etc.) on the left panel.

**Few-Shot Learning Tab**
1. Upload an example image + describe the expected answer.
2. Click **➕ Add Example** to add it to context.
3. Repeat to add more examples.
4. Upload a new image with a question and click **🚀 Generate**.
5. The model follows the pattern from your examples.

**Supported Inputs**
- 📷 Images: JPG, PNG, BMP, WebP, TIFF
- 🎤 Audio: WAV, MP3, FLAC, M4A (< 60 s)
- 📝 Text: Any language

**Tips**
- For image understanding, ask specific questions about the image content.
- For audio, the model can do ASR, speaker analysis, sound classification, etc.
- Multi-turn conversations maintain context automatically.
""")

    return demo
