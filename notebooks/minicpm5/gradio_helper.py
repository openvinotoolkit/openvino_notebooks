from threading import Thread
import queue
import re

import openvino_genai as ov_genai

MAX_NEW_TOKENS = 512

EXAMPLES = [
    ["Explain gravity to a 5-year-old."],
    ["Solve the equation: 2x + 5 = 15"],
    ["Write a 100-word blog post on the benefits of on-device AI."],
    ["Tell me a joke about calculus."],
    ["What are some common mistakes to avoid when writing code?"],
]


def format_reasoning(text: str) -> str:
    """
    Render the model's <think> section as an italic block.

    Params:
        text: partially generated answer that may contain <think>...</think> markers.
    Returns:
        Markdown-friendly text for the Gradio chat interface.
    """
    text = text.replace("<think>", "<small><i>🧠 Thinking...</i></small>\n\n")
    text = text.replace("</think>", "\n\n")
    text = re.sub(r"</small>\s*<small>", "", text)  # merge adjacent styling tags produced while streaming
    return text


def make_demo(pipe: ov_genai.LLMPipeline):
    """
    Create a Gradio chat interface for an OpenVINO GenAI pipeline.

    Params:
        pipe: LLMPipeline initialized with the converted MiniCPM5-2B model.
    Returns:
        Launched Gradio demo application.
    """
    import gradio as gr

    tokenizer = pipe.get_tokenizer()

    def bot(message: str, history: list, temperature: float, top_p: float, max_new_tokens: int | float, enable_thinking: bool):
        messages = list(history) + [{"role": "user", "content": message}]
        extra_context = None if enable_thinking else {"enable_thinking": False}
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, extra_context=extra_context)
        config = ov_genai.GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            apply_chat_template=False,
        )

        text_queue = queue.Queue()

        def stream_callback(subword: str) -> ov_genai.StreamingStatus:
            text_queue.put(subword)
            return ov_genai.StreamingStatus.RUNNING

        streamer = ov_genai.TextStreamer(tokenizer, stream_callback)

        def run_generation() -> None:
            try:
                pipe.generate(prompt, config, streamer=streamer)
                text_queue.put(None)
            except Exception as exc:  # noqa: BLE001
                text_queue.put(exc)

        thread = Thread(target=run_generation)
        thread.start()

        partial_text = ""
        while True:
            item = text_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                thread.join()
                raise item
            partial_text += item
            yield format_reasoning(partial_text)
        thread.join()

    demo = gr.ChatInterface(
        bot,
        title="MiniCPM5-2B 💬",
        description="Ask anything — enable or disable the model's thinking mode below.",
        examples=EXAMPLES,
        additional_inputs=[
            gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.05, label="Temperature"),
            gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p"),
            gr.Slider(minimum=64, maximum=2048, value=MAX_NEW_TOKENS, step=64, label="Max new tokens"),
            gr.Checkbox(value=True, label="Enable thinking"),
        ],
        additional_inputs_accordion=gr.Accordion("Generation parameters", open=False),
    )
    return demo
