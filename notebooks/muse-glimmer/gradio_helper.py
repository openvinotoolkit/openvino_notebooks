import html
import queue
import re
import time
from pathlib import Path
from threading import Thread
from typing import Union

import cv2
import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image

ANSWER_MARKERS = ("assistant to=user", "to=user")


def split_atem_response(text: str):
    """Split Muse Glimmer's decoded ATEM output into reasoning and final answer."""
    for marker in ANSWER_MARKERS:
        if marker in text:
            reasoning, answer = text.rsplit(marker, 1)
            reasoning = re.sub(r"^\s*to=self", "", reasoning).strip()
            if marker == "to=user":
                reasoning = reasoning.removesuffix("assistant").rstrip()
            return reasoning, answer.strip(), True

    if text.lstrip().startswith("to=self"):
        reasoning = re.sub(r"^\s*to=self", "", text).strip()
        return reasoning, "", False

    return "", text.strip(), False


def result_text(result) -> str:
    """Return text from VLMDecodedResults or a string-like result."""
    if hasattr(result, "texts") and result.texts:
        return result.texts[0]
    return str(result)


def generate_once(pipe, prompt, reasoning_strength="low", **kwargs) -> str:
    """Run one independent request with an explicit reasoning strength."""
    system_message = "You are a helpful AI assistant. Answer the user's request accurately.\n\n" f"Reasoning strength: {reasoning_strength}."
    pipe.start_chat(system_message=system_message)
    try:
        return result_text(pipe.generate(prompt, **kwargs))
    finally:
        pipe.finish_chat()


def display_atem_response(text: str, show_answer=True):
    """Display reasoning in a collapsed accordion and the answer as Markdown."""
    import ipywidgets as widgets
    from IPython.display import Markdown, display

    reasoning, answer, complete = split_atem_response(text)

    if reasoning:
        reasoning_view = widgets.HTML(f"<pre style='white-space:pre-wrap; margin:0'>{html.escape(reasoning)}</pre>")
        accordion = widgets.Accordion(children=[reasoning_view])
        accordion.set_title(0, "Reasoning")
        accordion.selected_index = None
        display(accordion)

    if show_answer and answer:
        display(Markdown(answer))
    elif show_answer and not complete:
        display(Markdown("⚠️ The generation stopped before the final `to=user` answer was produced."))

    return reasoning, answer


class ATEMDisplayStreamer:
    """Stream ATEM reasoning and answer channels into separate notebook outputs."""

    def __init__(self, show_answer=True):
        import ipywidgets as widgets
        from IPython.display import Markdown, display

        self.show_answer = show_answer
        self.raw_text = ""
        self.answer_started = False
        self._Markdown = Markdown

        self.status = widgets.HTML("<b>Generating:</b> waiting for the first token…")
        self.reasoning_view = widgets.HTML("<i>Waiting for reasoning…</i>")
        self.reasoning_accordion = widgets.Accordion(children=[self.reasoning_view])
        self.reasoning_accordion.set_title(0, "Reasoning (streaming)")
        self.reasoning_accordion.selected_index = 0

        display(self.status, self.reasoning_accordion)
        self.answer_handle = display(Markdown(""), display_id=True) if show_answer else None

    def __call__(self, subword):
        self.raw_text += subword
        self._render()
        # False tells OpenVINO GenAI to continue generation.
        return False

    def _render(self):
        reasoning, answer, complete = split_atem_response(self.raw_text)

        if complete:
            self.answer_started = True
            self.reasoning_view.value = f"<pre style='white-space:pre-wrap; margin:0'>{html.escape(reasoning)}</pre>"
            self.reasoning_accordion.set_title(0, "Reasoning")
            self.reasoning_accordion.selected_index = None
            self.status.value = f"<b>Generating answer…</b> {len(answer):,} characters"
            if self.answer_handle is not None:
                self.answer_handle.update(self._Markdown(answer))
        else:
            visible_reasoning = reasoning or self.raw_text
            self.reasoning_view.value = f"<pre style='white-space:pre-wrap; margin:0'>{html.escape(visible_reasoning)}</pre>"
            self.status.value = f"<b>Reasoning…</b> {len(visible_reasoning):,} characters generated"

    def finish(self, final_text=None):
        if final_text and len(final_text) > len(self.raw_text):
            self.raw_text = final_text
            self._render()

        reasoning, answer, complete = split_atem_response(self.raw_text)
        if complete:
            self.status.value = "✅ <b>Generation complete</b>"
        elif reasoning:
            self.status.value = "⚠️ <b>Generation stopped before the final answer.</b> " "Increase <code>max_new_tokens</code> and retry."
        else:
            self.status.value = "✅ <b>Generation complete</b>"
            if self.answer_handle is not None:
                self.answer_handle.update(self._Markdown(answer))
        return self.raw_text

    def fail(self, error):
        self.status.value = f"❌ <b>Generation failed:</b> {html.escape(str(error))}"


def generate_with_streaming(
    pipe,
    prompt,
    reasoning_strength="low",
    show_answer=True,
    **kwargs,
) -> str:
    """Run one request while streaming reasoning and answer into notebook widgets."""
    system_message = "You are a helpful AI assistant. Answer the user's request accurately.\n\n" f"Reasoning strength: {reasoning_strength}."
    display_streamer = ATEMDisplayStreamer(show_answer=show_answer)
    display_streamer.status.value = f"<b>Generating with {html.escape(reasoning_strength)} reasoning:</b> " "waiting for the first token…"
    result = None

    pipe.start_chat(system_message=system_message)
    try:
        result = pipe.generate(prompt, streamer=display_streamer, **kwargs)
    except Exception as error:
        display_streamer.fail(error)
        raise
    finally:
        pipe.finish_chat()

    final_text = result_text(result) if result is not None else None
    return display_streamer.finish(final_text)


class ChunkStreamer(ov_genai.StreamerBase):
    """Convert generated token IDs into iterable decoded text chunks."""

    def __init__(self, tokenizer, tokens_len=2):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_len = tokens_len
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration
        return value

    def write(self, token: Union[int, list[int]]) -> ov_genai.StreamingStatus:
        if isinstance(token, list):
            self.tokens_cache.extend(token)
        else:
            self.tokens_cache.append(token)

        if len(self.tokens_cache) % self.tokens_len:
            return self.get_stop_flag()

        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.print_len:
            self.text_queue.put(text[self.print_len :])
            self.print_len = len(text)
        return self.get_stop_flag()

    def get_stop_flag(self):
        return ov_genai.StreamingStatus.CANCEL if self._stop_flag else ov_genai.StreamingStatus.RUNNING

    def end(self):
        if self._ended:
            return
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.print_len:
            self.text_queue.put(text[self.print_len :])
        self.text_queue.put(None)
        self._stop_flag = True
        self._ended = True

    def reset(self):
        self.tokens_cache = []
        self.text_queue = queue.Queue()
        self.print_len = 0
        self._stop_flag = False
        self._ended = False


def load_video_frames(source, max_frames=8):
    """Load uniformly sampled BGR uint8 frames as [Frame, H, W, C]."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Cannot read frames from video: {source}")

    sample_count = min(max_frames, total_frames)
    indices = set(np.linspace(0, total_frames - 1, sample_count).astype(int))
    frames = []
    index = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if index in indices:
            frames.append(frame)
        index += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from: {source}")
    return ov.Tensor(np.stack(frames))


def make_demo(pipe, sample_image=None, sample_video=None):
    """Create a Muse Glimmer chat that renders reasoning separately from answers."""
    import gradio as gr

    max_new_tokens = 2048
    state = {"pipe": pipe}

    def bot(message, history, reasoning_strength):
        current_pipe = state["pipe"]
        history = history or []
        text = (message.get("text") or "").strip()
        files = message.get("files") or []

        images, videos = [], []
        for item in files:
            if isinstance(item, str):
                path = item
            elif isinstance(item, dict):
                path = item.get("path", "")
            else:
                path = getattr(item, "path", "")
            if str(path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                videos.append(path)
            elif path:
                images.append(path)

        if not history:
            system_message = "You are a helpful AI assistant. Answer the user's request accurately.\n\n" f"Reasoning strength: {reasoning_strength}."
            current_pipe.start_chat(system_message=system_message)

        for path in images:
            history.append({"role": "user", "content": gr.Image(path)})
        for path in videos:
            history.append({"role": "user", "content": gr.Video(path)})
        if text:
            history.append({"role": "user", "content": text})

        started_at = time.time()
        thinking_message = gr.ChatMessage(
            role="assistant",
            content="",
            metadata={"title": "🤔 Reasoning", "status": "pending"},
        )
        history.append(thinking_message)

        config = current_pipe.get_generation_config()
        config.max_new_tokens = max_new_tokens
        config.temperature = 1.0
        config.top_p = 0.95
        config.top_k = 64
        config.do_sample = True

        streamer = ChunkStreamer(current_pipe.get_tokenizer())

        def generate():
            try:
                image_tensors = [ov.Tensor(np.asarray(Image.open(path).convert("RGB"))) for path in images]
                video_tensors = [load_video_frames(path) for path in videos]
                generation_args = {
                    "generation_config": config,
                    "streamer": streamer,
                }
                if image_tensors:
                    generation_args["images"] = image_tensors
                if video_tensors:
                    generation_args["videos"] = video_tensors
                current_pipe.generate(text, **generation_args)
            except Exception as error:
                streamer.text_queue.put(f"\nGeneration error: {error}")
            finally:
                streamer.end()

        Thread(target=generate, daemon=True).start()

        raw_text = ""
        answer_started = False
        for new_text in streamer:
            raw_text += new_text
            reasoning, answer, complete = split_atem_response(raw_text)

            if complete:
                if not answer_started:
                    answer_started = True
                    if reasoning:
                        thinking_message.content = reasoning
                        thinking_message.metadata["status"] = "done"
                        thinking_message.metadata["duration"] = round(time.time() - started_at, 1)
                        history.append({"role": "assistant", "content": answer})
                    else:
                        history[-1] = {"role": "assistant", "content": answer}
                        thinking_message = None
                else:
                    history[-1]["content"] = answer
            elif thinking_message is not None:
                thinking_message.content = reasoning or raw_text

            yield gr.MultimodalTextbox(value=None), history, streamer

        if not answer_started:
            reasoning, answer, _ = split_atem_response(raw_text)
            if reasoning:
                thinking_message.content = reasoning
                thinking_message.metadata["status"] = "done"
                thinking_message.metadata["duration"] = round(time.time() - started_at, 1)
                history.append(
                    {
                        "role": "assistant",
                        "content": "The generation ended before the final answer. Increase `max_new_tokens` and retry.",
                    }
                )
            else:
                history[-1] = {"role": "assistant", "content": answer or raw_text}
            yield gr.MultimodalTextbox(value=None), history, streamer

    def stop(streamer):
        if streamer is not None:
            streamer.end()
        return None

    def clear(streamer):
        if streamer is not None:
            streamer.end()
        try:
            state["pipe"].finish_chat()
        except Exception:
            pass
        return [], None

    examples = []
    if sample_video and Path(sample_video).exists():
        examples.append([{"text": "Describe what is happening in this video.", "files": [str(sample_video)]}])
    if sample_image and Path(sample_image).exists():
        examples.append([{"text": "Describe this image in detail.", "files": [str(sample_image)]}])

    with gr.Blocks(fill_height=True) as demo:
        gr.Markdown("# Muse Glimmer-30B with OpenVINO")
        chatbot = gr.Chatbot(height="70vh")
        message = gr.MultimodalTextbox(
            file_types=["image", ".mp4"],
            file_count="multiple",
            placeholder="Ask a question and optionally attach images or video",
            show_label=False,
        )
        reasoning_strength = gr.Dropdown(
            choices=["low", "medium", "high", "xhigh"],
            value="high",
            label="Reasoning strength (applies when a new chat starts)",
        )
        streamer_state = gr.State(None)
        with gr.Row():
            stop_button = gr.Button("Stop")
            clear_button = gr.Button("Clear")

        if examples:
            gr.Examples(examples=examples, inputs=[message])

        message.submit(
            fn=bot,
            inputs=[message, chatbot, reasoning_strength],
            outputs=[message, chatbot, streamer_state],
        )
        stop_button.click(stop, inputs=[streamer_state], outputs=[streamer_state], queue=False)
        clear_button.click(
            clear,
            inputs=[streamer_state],
            outputs=[chatbot, streamer_state],
            queue=False,
        )

    return demo
