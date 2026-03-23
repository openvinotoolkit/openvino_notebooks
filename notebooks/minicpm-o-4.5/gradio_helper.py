"""
Gradio helper for MiniCPM-o 4.5 OpenVINO notebook demo  (Gradio ≥ 6.0).

Features
--------
  • Multimodal chat — text, image, and audio input
  • Streaming & non-streaming generation
  • Native thinking-mode display (reasoning_tags)
  • Few-shot learning tab
  • Sampling-parameter controls
  • Regenerate / Clear / Stop

Reference : https://github.com/OpenSQZ/MiniCPM-V-CookBook
Adapted for direct OpenVINO model calls (no client / server split).
"""

import re
import threading
import traceback
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import requests
from gradio.data_classes import FileData
from PIL import Image

# ───────────────────────────── Constants ─────────────────────────────────

MODEL_NAME = "MiniCPM-o 4.5 (OpenVINO)"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".wma", ".ogg", ".aac"}
ASSETS_DIR = Path(__file__).parent / "assets"

# ── Download example assets at import time (like qwen2-vl pattern) ─────────
_EXAMPLE_IMAGE_URLS = [
    (
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1d6a0188-5613-418d-a1fd-4560aae1d907",
        "example_bee.jpg",
    ),
    (
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/6cc7feeb-0721-4b5d-8791-2576ed9d2863",
        "example_baklava.png",
    ),
]
for _url, _fname in _EXAMPLE_IMAGE_URLS:
    _fp = Path(__file__).parent / _fname
    if not _fp.exists():
        try:
            print(f"Downloading example image: {_fname} …")
            Image.open(requests.get(_url, stream=True, timeout=30).raw).save(_fp)
        except Exception as _e:
            print(f"  Could not download {_fname}: {_e}")

# Note: Gradio 6 does not accept custom CSS via gr.Blocks(css=…).

# ───────────────────────────── Helpers ───────────────────────────────────


def _ftype(path: str) -> str:
    """Classify a file path as ``'image'``, ``'audio'``, or ``'unknown'``."""
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    return "unknown"


def _load_img(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_audio(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=16000, mono=True)
    return y


def _strip_think(text: str) -> str:
    """Remove ``<think>…</think>`` blocks (used when re-feeding history)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _clean_tts(text: str) -> str:
    """Strip TTS markers that may leak from the decoder."""
    return text.replace("<|tts_eos|>", "").replace("<|tts_bos|>", "")


def _reset(model) -> None:
    """Best-effort reset of model / KV-cache state."""
    for fn in ("reset_state", "reset_session"):
        if hasattr(model, fn):
            try:
                getattr(model, fn)()
            except Exception:  # nosec B110 - best-effort KV-cache reset, non-critical
                pass
            return


def _resolve_content_item(raw, role: str):
    """Resolve a single content element into a model-ready item.

    Handles all formats that Gradio 6 Chatbot may produce after its
    postprocess → preprocess round-trip:

    * Plain ``str``
    * ``FileData`` object
    * ``{'type': 'text', 'text': '...'}``  (TextMessage dict)
    * ``{'type': 'file', 'file': {'path': '...'}}``  (FileMessage dict)
    * A ``list`` of the above  (Gradio 6 wraps every content in a list)
    """
    # ── list of sub-items → recurse and collect ─────────────────────
    if isinstance(raw, list):
        items = []
        for sub in raw:
            r = _resolve_content_item(sub, role)
            if r is not None:
                items.append(r)
        return items if items else None

    # ── FileData object (from _user before round-trip) ─────────────
    if isinstance(raw, FileData):
        return _load_file(raw.path)

    # ── plain string ───────────────────────────────────────────────
    if isinstance(raw, str):
        text = _strip_think(raw) if role == "assistant" else raw.strip()
        return text if text else None

    # ── dict produced by Gradio 6 chatbot preprocess ───────────────
    if isinstance(raw, dict):
        tp = raw.get("type", "")
        if tp == "text":
            text = (raw.get("text") or "").strip()
            if role == "assistant":
                text = _strip_think(text)
            return text if text else None
        if tp == "file":
            finfo = raw.get("file") or {}
            path = finfo.get("path") or finfo.get("url") or ""
            if path:
                return _load_file(path)
        return None

    return None


def _load_file(path: str):
    """Load an image or audio file into the format expected by ``ov_model.chat()``."""
    ft = _ftype(path)
    if ft == "image":
        return _load_img(path)
    if ft == "audio":
        return _load_audio(path)
    return None


def _history_to_msgs(
    history: list[dict],
    sys_prompt: str = "",
) -> list[dict]:
    """Convert Gradio 6 chatbot history → ``ov_model.chat()`` message list.

    After the Chatbot postprocess→preprocess round-trip in Gradio 6 each
    entry looks like::

        {'role': 'user',
         'content': [{'type': 'text', 'text': '...'}, ...],
         'metadata': {...}, 'options': [...]}

    This function normalises that back to the simple format that
    ``ov_model.chat()`` expects.
    """
    msgs: list[dict] = []
    if sys_prompt and sys_prompt.strip():
        msgs.append({"role": "system", "content": sys_prompt.strip()})

    for entry in history:
        role = entry.get("role", "user")
        raw = entry.get("content", "")

        # Skip thinking-display entries produced by reasoning_tags
        meta = entry.get("metadata")
        if meta and isinstance(meta, dict) and meta.get("title"):
            continue

        # Resolve content (may be str, FileData, dict, or list of those)
        resolved = _resolve_content_item(raw, role)
        if resolved is None:
            continue

        # Flatten single-element lists
        if isinstance(resolved, list) and len(resolved) == 1:
            resolved = resolved[0]

        # Merge consecutive same-role entries into a multimodal list
        if msgs and msgs[-1]["role"] == role:
            prev = msgs[-1]["content"]
            if isinstance(resolved, list):
                msgs[-1]["content"] = [*prev, *resolved] if isinstance(prev, list) else [prev, *resolved]
            else:
                msgs[-1]["content"] = [*prev, resolved] if isinstance(prev, list) else [prev, resolved]
        else:
            msgs.append({"role": role, "content": resolved})

    return msgs


# ──────────────────────────── make_demo ──────────────────────────────────


def make_demo(ov_model):
    """Build and return a ``gr.Blocks`` demo wired to *ov_model*."""

    stop_ev = threading.Event()

    # ── Chat handlers ──────────────────────────────────────────────────

    def _user(msg, hist):
        """Append user turn (files first, then text) to *hist*."""
        print(f"[user] message={msg}")
        if msg is None:
            return hist, gr.MultimodalTextbox(value=None)
        text = (msg.get("text") or "").strip()
        files = msg.get("files") or []
        if not text and not files:
            return hist, gr.MultimodalTextbox(value=None)

        for f in files:
            if isinstance(f, str):
                fp = f
            elif isinstance(f, dict):
                fp = f.get("path", "")
            else:
                fp = ""
            if fp:
                hist.append({"role": "user", "content": FileData(path=fp)})
        if text:
            hist.append({"role": "user", "content": text})

        return hist, gr.MultimodalTextbox(value=None)

    def _bot(hist, think, stream, max_tok, temp, tp, tk, rp, sys_prompt):
        """Generate an assistant response (streaming or blocking)."""
        print(f"[bot] history_len={len(hist)}, think={think}, stream={stream}")
        if not hist:
            print("[bot] empty history, skipping")
            yield hist
            return

        stop_ev.clear()

        # Build model messages
        try:
            msgs = _history_to_msgs(hist, sys_prompt)
        except Exception as exc:
            print(f"[bot] _history_to_msgs error: {exc}")
            traceback.print_exc()
            hist.append({"role": "assistant", "content": f"⚠️ {exc}"})
            yield hist
            return

        print(f"[bot] msgs count={len(msgs)}, last_role={msgs[-1].get('role') if msgs else 'N/A'}")
        for i, m in enumerate(msgs):
            c = m["content"]
            ctype = type(c).__name__ if not isinstance(c, list) else f"list[{len(c)}]"
            preview = str(c)[:80] if isinstance(c, str) else ctype
            print(f"  msg[{i}] role={m['role']} content={preview}")

        if not msgs or msgs[-1].get("role") != "user":
            print("[bot] no user message found, skipping")
            yield hist
            return

        _reset(ov_model)

        kw = dict(
            msgs=msgs,
            max_new_tokens=int(max_tok),
            do_sample=(temp > 0),
            temperature=max(float(temp), 0.01),
            top_p=float(tp),
            top_k=int(tk),
            repetition_penalty=float(rp),
            enable_thinking=bool(think),
            stream=bool(stream),
        )

        try:
            if stream:
                streamer = ov_model.chat(**kw)
                hist.append({"role": "assistant", "content": ""})
                for chunk in streamer:
                    if stop_ev.is_set():
                        break
                    hist[-1]["content"] += _clean_tts(chunk)
                    yield hist
            else:
                ans = ov_model.chat(**kw)
                if isinstance(ans, str):
                    ans = _clean_tts(ans)
                hist.append({"role": "assistant", "content": ans})
                yield hist
        except Exception:
            hist.append(
                {
                    "role": "assistant",
                    "content": f"⚠️ Generation error:\n```\n{traceback.format_exc()}\n```",
                }
            )
            yield hist

    def _stop():
        stop_ev.set()

    def _regen(hist, *args):
        """Drop last assistant turn, then re-generate."""
        while hist and hist[-1].get("role") == "assistant":
            hist.pop()
        if not hist:
            yield hist
            return
        yield from _bot(hist, *args)

    def _clear():
        _reset(ov_model)
        return []

    # ── Few-shot handlers ──────────────────────────────────────────────

    def _fs_fmt(st):
        if not st:
            return "*No examples yet.*"
        return "\n\n---\n\n".join(
            f"**Example {i}** {'🖼️' if e.get('image') else ''}\n" f"- **User:** {e['user']}\n- **Asst:** {e['assistant']}" for i, e in enumerate(st, 1)
        )

    def _fs_add(st, img, utxt, atxt):
        if not (utxt or "").strip():
            gr.Warning("User text is required.")
            return st, _fs_fmt(st)
        st = list(st or [])
        st.append(
            {
                "image": img,
                "user": utxt.strip(),
                "assistant": (atxt or "").strip(),
            }
        )
        return st, _fs_fmt(st)

    def _fs_clear():
        return [], _fs_fmt([])

    def _fs_gen(st, qi, qt, think, mt, tmp, tp, tk, rp):
        if not (qt or "").strip():
            return "Please enter a query first."
        stop_ev.clear()

        msgs: list[dict] = []
        for e in st or []:
            c = [_load_img(e["image"]), e["user"]] if e.get("image") else e["user"]
            msgs.append({"role": "user", "content": c})
            if e.get("assistant"):
                msgs.append({"role": "assistant", "content": e["assistant"]})

        qc = [_load_img(qi), qt.strip()] if qi else qt.strip()
        msgs.append({"role": "user", "content": qc})

        _reset(ov_model)

        try:
            r = ov_model.chat(
                msgs=msgs,
                max_new_tokens=int(mt),
                do_sample=(tmp > 0),
                temperature=max(float(tmp), 0.01),
                top_p=float(tp),
                top_k=int(tk),
                repetition_penalty=float(rp),
                enable_thinking=bool(think),
                stream=False,
            )
            return _clean_tts(r) if isinstance(r, str) else str(r)
        except Exception:
            return f"⚠️ Error:\n```\n{traceback.format_exc()}\n```"

    # ── Chatbot example presets ────────────────────────────────────────

    chat_examples = [
        {"text": "Hello! What can you do?", "display_text": "👋 Say hello"},
    ]
    _audio = ASSETS_DIR / "system_ref_audio.wav"
    if _audio.exists():
        chat_examples.append(
            {
                "text": "Please describe what you hear in this audio.",
                "files": [str(_audio)],
                "display_text": "🎵 Describe audio",
            }
        )

    # ── Build Blocks ───────────────────────────────────────────────────

    with gr.Blocks(title=MODEL_NAME, fill_height=True) as demo:
        gr.Markdown(f"# 🤖 {MODEL_NAME}")

        with gr.Tabs():
            # ═══════════════════ Chat ═══════════════════════════════════
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(
                    height=520,
                    reasoning_tags=[("<think>", "</think>")],
                    examples=chat_examples,
                )

                msg = gr.MultimodalTextbox(
                    placeholder="Type a message or attach images / audio …",
                    show_label=False,
                    submit_btn="Send",
                    stop_btn="Stop",
                    file_count="multiple",
                    sources=["upload", "microphone"],
                )

                with gr.Accordion("⚙️ Settings", open=False):
                    with gr.Row():
                        c_think = gr.Checkbox(label="Thinking Mode", value=False)
                        c_stream = gr.Checkbox(label="Streaming", value=True)
                    with gr.Row():
                        c_tok = gr.Slider(
                            64,
                            4096,
                            value=2048,
                            step=64,
                            label="Max Tokens",
                        )
                        c_temp = gr.Slider(
                            0.0,
                            2.0,
                            value=0.7,
                            step=0.05,
                            label="Temperature",
                        )
                    with gr.Row():
                        c_tp = gr.Slider(
                            0.0,
                            1.0,
                            value=0.8,
                            step=0.05,
                            label="Top-P",
                        )
                        c_tk = gr.Slider(1, 200, value=100, step=1, label="Top-K")
                    with gr.Row():
                        c_rp = gr.Slider(
                            1.0,
                            2.0,
                            value=1.05,
                            step=0.01,
                            label="Repetition Penalty",
                        )
                    c_sys = gr.Textbox(
                        label="System Prompt",
                        placeholder="Optional system prompt …",
                        lines=2,
                    )

                # ── Clickable examples (image + text / audio + text) ─────
                _ex_rows: list[list] = []
                _img1 = Path(__file__).parent / "example_bee.jpg"
                _img2 = Path(__file__).parent / "example_baklava.png"
                _audio_ex = ASSETS_DIR / "system_ref_audio.wav"
                if _img1.exists():
                    _ex_rows.append([{"text": "What is on the flower? Describe in detail.", "files": [str(_img1)]}])
                if _img2.exists():
                    _ex_rows.append([{"text": "How do you make this pastry?", "files": [str(_img2)]}])
                if _audio_ex.exists():
                    _ex_rows.append([{"text": "Please describe what you hear in this audio.", "files": [str(_audio_ex)]}])
                _ex_rows.append([{"text": "Hello! What multimodal tasks can you handle?"}])
                gr.Examples(
                    examples=_ex_rows,
                    inputs=[msg],
                    label="✨ Quick examples — click to fill, then press Send",
                )

                with gr.Row():
                    btn_regen = gr.Button("🔄 Regenerate", variant="secondary")
                    btn_clear = gr.Button("🗑️ Clear", variant="secondary")

                # Wire events
                gen = [
                    c_think,
                    c_stream,
                    c_tok,
                    c_temp,
                    c_tp,
                    c_tk,
                    c_rp,
                    c_sys,
                ]
                msg.submit(
                    _user,
                    [msg, chatbot],
                    [chatbot, msg],
                ).then(
                    _bot,
                    [chatbot] + gen,
                    chatbot,
                )
                # Stop button (built into MultimodalTextbox)
                msg.stop(_stop, [], [])

                btn_regen.click(_regen, [chatbot] + gen, chatbot)
                btn_clear.click(_clear, [], chatbot)

            # ═══════════════════ Few-Shot ═══════════════════════════════
            with gr.Tab("📚 Few-Shot"):
                gr.Markdown(
                    "### Few-Shot Prompting\n" "Add image + text examples, then query the model.",
                )
                fs_st = gr.State([])
                with gr.Row():
                    with gr.Column():
                        fs_img = gr.Image(
                            label="Example Image (optional)",
                            type="filepath",
                        )
                        fs_utxt = gr.Textbox(label="User Text", lines=2)
                        fs_atxt = gr.Textbox(label="Assistant Text", lines=2)
                        with gr.Row():
                            fs_add = gr.Button("➕ Add Example", variant="primary")
                            fs_clr = gr.Button("🗑️ Clear All")
                    with gr.Column():
                        fs_disp = gr.Markdown("*No examples yet.*")

                gr.Markdown("---\n### Query")
                with gr.Row():
                    with gr.Column():
                        fq_img = gr.Image(
                            label="Query Image (optional)",
                            type="filepath",
                        )
                        fq_txt = gr.Textbox(label="Query Text", lines=2)
                    with gr.Column():
                        fq_out = gr.Textbox(
                            label="Model Response",
                            lines=6,
                            interactive=False,
                        )

                with gr.Accordion("⚙️ Settings", open=False):
                    with gr.Row():
                        f_think = gr.Checkbox(label="Thinking", value=False)
                        f_tok = gr.Slider(
                            64,
                            4096,
                            value=2048,
                            step=64,
                            label="Max Tokens",
                        )
                    with gr.Row():
                        f_temp = gr.Slider(
                            0.0,
                            2.0,
                            value=0.7,
                            step=0.05,
                            label="Temperature",
                        )
                        f_tp = gr.Slider(
                            0.0,
                            1.0,
                            value=0.8,
                            step=0.05,
                            label="Top-P",
                        )
                    with gr.Row():
                        f_tk = gr.Slider(1, 200, value=100, step=1, label="Top-K")
                        f_rp = gr.Slider(
                            1.0,
                            2.0,
                            value=1.05,
                            step=0.01,
                            label="Repetition Penalty",
                        )

                fq_btn = gr.Button("🚀 Generate", variant="primary")

                fs_add.click(
                    _fs_add,
                    [fs_st, fs_img, fs_utxt, fs_atxt],
                    [fs_st, fs_disp],
                )
                fs_clr.click(_fs_clear, [], [fs_st, fs_disp])
                fq_btn.click(
                    _fs_gen,
                    [fs_st, fq_img, fq_txt, f_think, f_tok, f_temp, f_tp, f_tk, f_rp],
                    fq_out,
                )

            # ═══════════════════ How to Use ═════════════════════════════
            with gr.Tab("ℹ️ How to Use"):
                gr.Markdown(
                    """
## Usage Guide

### 💬 Chat

| Action | How |
|--------|-----|
| Text | Type your question and press **Send** |
| Image | Click 📎 to upload JPG / PNG / BMP / WebP |
| Audio | Click 📎 to upload WAV / MP3 / FLAC, or use 🎙️ microphone |
| Stop | Click **Stop** to interrupt generation |
| Thinking | Toggle **Thinking Mode** in ⚙️ Settings to see chain-of-thought |

### 📚 Few-Shot

1. Add one or more image + text examples with expected responses.
2. Enter a query (optionally with an image).
3. Click **Generate** — the model uses the examples as in-context demos.

### ⚙️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Max Tokens | 2048 | Maximum output length |
| Temperature | 0.7 | Randomness (0 = greedy) |
| Top-P | 0.8 | Nucleus sampling threshold |
| Top-K | 100 | Top-K filtering |
| Repetition Penalty | 1.05 | Penalise repeated tokens |
| System Prompt | — | Optional instruction prepended to conversation |
"""
                )

    return demo
