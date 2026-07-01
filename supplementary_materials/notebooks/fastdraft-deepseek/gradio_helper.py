import openvino as ov
import openvino_genai as ov_genai
from uuid import uuid4
from threading import Event, Thread
import queue
import sys
import time

from genai_helper import ChunkStreamer

core = ov.Core()


english_examples = [
    ["Which is bigger, 9.9 or 9.11?"],
    ["Classify the following numbers as 'prime' or 'composite' - 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16.?"],
    ["What are the classifications of Academic Degrees?"],
    ["Which word does not belong to the other: Hammer, Screwdriver, Nail, Wood"],
    ["Identify which instrument is string or percussion: Kpanlogo, Shamisen"],
    ["Which of the following are colors: red, black, yellow, orange, sun, sunflower, chips, book, white, pink, blue, keyboard."],
]


def make_demo(pipe, stop_strings, model_id):
    import gradio as gr

    streamer = ChunkStreamer(pipe.tokenizer, tokens_len=6)
    conversation = []

    def get_uuid():
        """
        universal unique identifier for thread
        """
        return str(uuid4())

    def apply_format(partial_text: str):
        """
        helper for updating partially generated answer, used by default

        Params:
        partial_text: text buffer for storing previosly generated text
        new_text: text update for the current step
        Returns:
        updated text string

        """
        blockquote_style = """font-size: 10px;
background: #e4e4e4;
border-left: 10px solid #ccc; 
margin: 0.5em 30px;
padding: 0.5em 10px;
color: black;"""
        summary_style = """font-weight: bold;
font-size: 14px;
list-style-position: outside;
margin: 0.5em 15px;
padding: 0px 0px 10px 15px;"""
        formatted_text = ""
        splits = partial_text.split("</think>")
        for i, s in enumerate(splits):
            formatted_text += (
                s.replace(
                    "<think>",
                    f'<details {"open" if i == (len(splits) - 1) else ""} style="margin:0px;padding:0px;"><summary style="{summary_style}">Thought</summary><blockquote style="{blockquote_style}"><p>',
                )
                + "</p></blockquote></details>"
            )
        return formatted_text

    def is_partial_stop(output, stop_str):
        """Check whether the output contains a partial stop str."""
        for i in range(0, min(len(output), len(stop_str))):
            if stop_str.startswith(output[-i:]):
                return True
        return False

    def bot(message, history, temperature, top_p, top_k, num_assistant_tokens, repetition_penalty, max_tokens):
        """
        callback function for running chatbot on submit button click

        Params:
        message: new message from user
        history: conversation history
        temperature:  parameter for control the level of creativity in AI-generated text.
                        By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
        top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
        top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
        repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
        active_chat: chat state, if true then chat is running, if false then we should start it here.
        Returns:
        message: reset message and make it ""
        history: updated history with message and answer from chatbot
        active_chat: if we are here, the chat is running or will be started, so return True
        """
        config = pipe.get_generation_config()
        config.temperature = temperature
        config.top_p = top_p
        config.top_k = top_k
        config.do_sample = temperature > 0.0
        config.max_new_tokens = max_tokens
        config.repetition_penalty = repetition_penalty
        config.num_assistant_tokens = num_assistant_tokens
        config.include_stop_str_in_output = True
        config.stop_strings = set(stop_strings)
        history = history or []
        history.append([message, ""])
        conversation.append({"role": "user", "content": message})
        stream_complete = Event()

        def generate_and_signal_complete():
            """
            genration function for single thread
            """
            streamer.reset()
            pipe.generate(conversation, config, streamer, apply_chat_template=True)
            stream_complete.set()
            streamer.end()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()
        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            pos = -1
            for s in config.stop_strings:
                if (pos := partial_text.rfind(s)) != -1:
                    break
            if pos != -1:
                partial_text = partial_text[:pos]
                history[-1][1] = apply_format(partial_text)
                yield "", history
                break
            elif any([is_partial_stop(partial_text, s) for s in config.stop_strings]):
                continue
            history[-1][1] = apply_format(partial_text)
            yield "", history

        t1.join()
        conversation.append({"role": "assistant", "content": partial_text})
        print(conversation)
        return "", history

    def clear_history():
        nonlocal conversation
        conversation = []
        return None

    examples = english_examples

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        conversation_id = gr.State(get_uuid)
        gr.Markdown(f"""<h1><center>OpenVINO {model_id} + FastDraft Chatbot</center></h1>""")
        chatbot = gr.Chatbot(height=500)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                    container=False,
                )
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    clear = gr.Button("Clear")
        with gr.Row():
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
                                minimum=0.0,
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
                                value=0,
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
                                value=1.0,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )
                    with gr.Column():
                        with gr.Row():
                            num_assistant_tokens = gr.Slider(
                                label="Number of assistant tokens",
                                value=3,
                                minimum=1,
                                maximum=8,
                                step=1,
                                interactive=True,
                                info="Number of tokens for the draft to predict for speculative decoding",
                                visible=True,
                            )
                    with gr.Column():
                        with gr.Row():
                            max_tokens = gr.Slider(
                                label="Max new tokens",
                                value=1024,
                                minimum=8,
                                maximum=2048,
                                step=32,
                                interactive=True,
                                info=("Maximum new tokens added to answer. Higher value can work for long response, but require more time to complete"),
                            )
        gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")

        msg.submit(
            fn=bot,
            inputs=[msg, chatbot, temperature, top_p, top_k, num_assistant_tokens, repetition_penalty, max_tokens],
            outputs=[msg, chatbot],
            queue=True,
        )
        submit.click(
            fn=bot,
            inputs=[msg, chatbot, temperature, top_p, top_k, num_assistant_tokens, repetition_penalty, max_tokens],
            outputs=[msg, chatbot],
            queue=True,
        )
        clear.click(fn=clear_history, outputs=[chatbot], queue=False)

        return demo
