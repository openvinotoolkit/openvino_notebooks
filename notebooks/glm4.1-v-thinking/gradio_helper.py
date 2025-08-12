import copy
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
import fitz
import gradio as gr
import time
import html
import torch
from transformers import TextIteratorStreamer

stop_generation = False


class GLM4VModel:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def _strip_html(self, text: str) -> str:
        return re.sub(r"<[^>]+>", "", text).strip()

    def _wrap_text(self, text: str):
        return [{"type": "text", "text": text}]

    def _pdf_to_imgs(self, pdf_path):
        doc = fitz.open(pdf_path)
        imgs = []
        for i in range(doc.page_count):
            pix = doc.load_page(i).get_pixmap(dpi=180)
            img_p = os.path.join(tempfile.gettempdir(), f"{Path(pdf_path).stem}_{i}.png")
            pix.save(img_p)
            imgs.append(img_p)
        doc.close()
        return imgs

    def _ppt_to_imgs(self, ppt_path):
        tmp = tempfile.mkdtemp()
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", tmp, ppt_path],
            check=True,
        )
        pdf_path = os.path.join(tmp, Path(ppt_path).stem + ".pdf")
        return self._pdf_to_imgs(pdf_path)

    def _files_to_content(self, media):
        out = []
        for f in media or []:
            ext = Path(f.name).suffix.lower()
            if ext in [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg", ".m4v"]:
                out.append({"type": "video", "url": f.name})
            elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
                out.append({"type": "image", "url": f.name})
            elif ext in [".ppt", ".pptx"]:
                for p in self._ppt_to_imgs(f.name):
                    out.append({"type": "image", "url": p})
            elif ext == ".pdf":
                for p in self._pdf_to_imgs(f.name):
                    out.append({"type": "image", "url": p})
        return out

    def _stream_fragment(self, buf: str, skip_think: bool = False):
        think_html = ""
        if "<think>" in buf and not skip_think:
            if "</think>" in buf:
                seg = re.search(r"<think>(.*?)</think>", buf, re.DOTALL)
                if seg:
                    think_content = seg.group(1).strip().replace("\\n", "\n").replace("\n", "<br>")
                    think_html = (
                        "<details open><summary style='cursor:pointer;font-weight:bold;color:#007acc;'>💭 Thinking</summary>"
                        "<div style='color:#555555;line-height:1.6;padding:15px;border-left:4px solid #007acc;margin:10px 0;background-color:#f0f7ff;border-radius:4px;'>"
                        + think_content
                        + "</div></details>"
                    )
            else:
                part = buf.split("<think>", 1)[1]
                think_content = part.replace("\\n", "\n").replace("\n", "<br>")
                think_html = (
                    "<details open><summary style='cursor:pointer;font-weight:bold;color:#007acc;'>💭 Thinking</summary>"
                    "<div style='color:#555555;line-height:1.6;padding:15px;border-left:4px solid #007acc;margin:10px 0;background-color:#f0f7ff;border-radius:4px;'>"
                    + think_content
                    + "</div></details>"
                )

        answer_html = ""
        if "<answer>" in buf:
            if "</answer>" in buf:
                seg = re.search(r"<answer>(.*?)</answer>", buf, re.DOTALL)
                if seg:
                    answer_html = seg.group(1).strip()
            else:
                answer_html = buf.split("<answer>", 1)[1]

        if answer_html:
            answer_html_raw = answer_html.replace("\\n", "\n")
            if "<" in answer_html_raw and ">" in answer_html_raw:
                escaped = html.escape(answer_html_raw)
                answer_html = f"<pre class='code-block'><code class='language-html'>{escaped}</code></pre>"
            else:
                answer_html = f"<div style='margin:0.5em 0; white-space: pre-wrap; line-height:1.6;'>{html.escape(answer_html_raw)}</div>"

        if not think_html and not answer_html:
            return self._strip_html(buf)
        return think_html + answer_html

    def _build_messages(self, raw_hist, sys_prompt):
        msgs = []
        if sys_prompt.strip():
            msgs.append({"role": "system", "content": [{"type": "text", "text": sys_prompt.strip()}]})
        for h in raw_hist:
            if h["role"] == "user":
                msgs.append({"role": "user", "content": h["content"]})
            else:
                raw = re.sub(r"<think>.*?</think>", "", h["content"], flags=re.DOTALL)
                raw = re.sub(r"<details.*?</details>", "", raw, flags=re.DOTALL)
                msgs.append({"role": "assistant", "content": self._wrap_text(self._strip_html(raw).strip())})
        return msgs

    def stream_generate(self, raw_hist, sys_prompt: str):
        global stop_generation
        stop_generation = False
        msgs = self._build_messages(raw_hist, sys_prompt)
        inputs = self.processor.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=False)
        gen_kwargs = dict(
            inputs,
            max_new_tokens=8192,
            repetition_penalty=1.1,
            do_sample=True,
            top_k=2,
            temperature=0.01,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        buf = ""
        for tok in streamer:
            if stop_generation:
                break
            buf += tok
            yield self._stream_fragment(buf)
        thread.join()


def format_display_content(content):
    if isinstance(content, list):
        text_parts = []
        file_count = 0
        for item in content:
            if item["type"] == "text":
                text_parts.append(item["text"])
            else:
                file_count += 1
        display_text = " ".join(text_parts)
        if file_count > 0:
            return f"[{file_count} file(s) uploaded]\n{display_text}"
        return display_text
    return content


def create_display_history(raw_hist):
    display_hist = []
    for h in raw_hist:
        if h["role"] == "user":
            display_content = format_display_content(h["content"])
            display_hist.append({"role": "user", "content": display_content})
        else:
            display_hist.append({"role": "assistant", "content": h["content"]})
    return display_hist


def check_files(files):
    vids = imgs = ppts = pdfs = 0
    for f in files or []:
        ext = Path(f.name).suffix.lower()
        if ext in [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg", ".m4v"]:
            vids += 1
        elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
            imgs += 1
        elif ext in [".ppt", ".pptx"]:
            ppts += 1
        elif ext == ".pdf":
            pdfs += 1
    if vids > 1 or ppts > 1 or pdfs > 1:
        return False, "Only one video or one PPT or one PDF allowed"
    if imgs > 10:
        return False, "Maximum 10 images allowed"
    if (ppts or pdfs) and (vids or imgs) or (vids and imgs):
        return False, "Cannot mix documents, videos, and images"
    return True, ""


def reset():
    global stop_generation
    stop_generation = True
    time.sleep(0.1)
    return [], [], None, ""


def make_demo(model, processor):
    glm4v = GLM4VModel(model, processor)

    def chat(files, msg, raw_hist, sys_prompt):
        global stop_generation
        stop_generation = False
        ok, err = check_files(files)
        if not ok:
            raw_hist.append({"role": "assistant", "content": err})
            display_hist = create_display_history(raw_hist)
            yield display_hist, copy.deepcopy(raw_hist), None, ""
            return
        payload = glm4v._files_to_content(files) if files else None
        if msg.strip():
            if payload is None:
                payload = glm4v._wrap_text(msg.strip())
            else:
                payload.append({"type": "text", "text": msg.strip()})
        user_rec = {"role": "user", "content": payload if payload else msg.strip()}
        if raw_hist is None:
            raw_hist = []
        raw_hist.append(user_rec)
        place = {"role": "assistant", "content": ""}
        raw_hist.append(place)
        display_hist = create_display_history(raw_hist)
        yield display_hist, copy.deepcopy(raw_hist), None, ""
        for chunk in glm4v.stream_generate(raw_hist[:-1], sys_prompt):
            if stop_generation:
                break
            place["content"] = chunk
            display_hist = create_display_history(raw_hist)
            yield display_hist, copy.deepcopy(raw_hist), None, ""
        display_hist = create_display_history(raw_hist)
        yield display_hist, copy.deepcopy(raw_hist), None, ""

    demo = gr.Blocks(title="GLM-4.1V-9B-Thinking-OpenVINO", theme=gr.themes.Soft())

    with demo:
        gr.Markdown("<div style='text-align:center;font-size:32px;font-weight:bold;margin-bottom:20px;'>GLM-4.1V-9B-Thinking-OpenVINO</div>")
        raw_history = gr.State([])

        with gr.Row():
            with gr.Column(scale=7):
                chatbox = gr.Chatbot(label="Chat", type="messages", height=600, elem_classes="chatbot-container", sanitize_html=False, line_breaks=True)
                textbox = gr.Textbox(label="Message", lines=3)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(scale=3):
                up = gr.File(label="Upload Files", file_count="multiple", file_types=["file"], type="filepath")
                gr.Markdown("Supports images / videos / PPT / PDF")
                gr.Markdown(
                    "The maximum supported input is 10 images or 1 video/PPT/PDF(less than 10 pages) in this demo. "
                    "During the conversation, video and images cannot be present at the same time."
                )
                sys = gr.Textbox(label="System Prompt", lines=6)

        send.click(chat, inputs=[up, textbox, raw_history, sys], outputs=[chatbox, raw_history, up, textbox])
        textbox.submit(chat, inputs=[up, textbox, raw_history, sys], outputs=[chatbox, raw_history, up, textbox])
        clear.click(reset, outputs=[chatbox, raw_history, up, textbox])
    return demo
