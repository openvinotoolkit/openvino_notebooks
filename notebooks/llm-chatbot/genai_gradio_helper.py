import re
from pathlib import Path
from typing import Any
import numpy as np
from queue import Queue
import openvino_tokenizers
from openvino_genai import StreamerBase
import openvino as ov
from uuid import uuid4
from threading import Event, Thread

max_new_tokens = 256

core = ov.Core()

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

class TextQueue:
    def __init__(self) -> None:
        self.text_queue = Queue()
        self.stop_signal = None
        self.stop_tokens = []

    def __call__(self, text) -> Any:
        self.text_queue.put(text)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value == self.stop_signal or value in self.stop_tokens:
            raise StopIteration()
        else:
            return value

    def reset(self):
        self.text_queue = Queue()

    def end(self):
        self.text_queue.put(self.stop_signal)


def get_gradio_helper(pipe, model_configuration, model_id, model_language):
    import gradio as gr

    max_new_tokens = 256

    start_message = model_configuration["start_message"]
    history_template = model_configuration.get("history_template")
    current_message_template = model_configuration.get("current_message_template")


    def get_uuid():
        """
        universal unique identifier for thread
        """
        return str(uuid4())


    def convert_history_to_input(history):
        """
        function for conversion history stored as list pairs of user and assistant messages to tokens according to model expected conversation template
        Params:
        history: dialogue history
        Returns:
        history in token format
        """
        new_prompt = f"{start_message}"
        if history_template is None:
            for user_msg, model_msg in history:
                new_prompt += user_msg + "\n" + model_msg + "\n"
            return new_prompt
        else:
            new_prompt = "".join(["".join([history_template.format(num=round, user=item[0], assistant=item[1])]) for round, item in enumerate(history[:-1])])
            new_prompt += "".join(
                [
                    "".join(
                        [
                            current_message_template.format(
                                num=len(history) + 1,
                                user=history[-1][0],
                                assistant=history[-1][1],
                            )
                        ]
                    )
                ]
            )

        return new_prompt


    def default_partial_text_processor(partial_text: str, new_text: str):
        """
        helper for updating partially generated answer, used by default

        Params:
        partial_text: text buffer for storing previosly generated text
        new_text: text update for the current step
        Returns:
        updated text string

        """
        partial_text += new_text
        return partial_text


    text_processor = model_configuration.get("partial_text_processor", default_partial_text_processor)


    def bot(message, history, temperature, top_p, top_k, repetition_penalty):
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
        streamer = TextQueue()
        config = pipe.get_generation_config()
        config.temperature = temperature
        config.top_p = top_p
        config.top_k = top_k
        config.do_sample = temperature > 0.0
        config.max_new_tokens = max_new_tokens
        config.repetition_penalty = repetition_penalty

        # history = [['message', 'chatbot answer'], ...]
        history.append([message, ""])
        new_prompt = convert_history_to_input(history)

        stream_complete = Event()

        def generate_and_signal_complete():
            """
            genration function for single thread
            """
            streamer.reset()
            pipe.generate(new_prompt, config, streamer)
            stream_complete.set()
            streamer.end()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        partial_text = ""
        for new_text in streamer:
            partial_text = text_processor(partial_text, new_text)
            history[-1][1] = partial_text
            yield "", history, streamer


    def stop_chat(streamer):
        if streamer is not None:
            streamer.end()
        return None


    def stop_chat_and_clear_history(streamer):
        if streamer is not None:
            streamer.end()
        return None, None

    examples = chinese_examples if (model_language == "Chinese") else japanese_examples if (model_language == "Japanese") else english_examples


    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        streamer = gr.State(None)
        conversation_id = gr.State(get_uuid)
        gr.Markdown(f"""<h1><center>OpenVINO {model_id} Chatbot</center></h1>""")
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
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
        with gr.Row():
            with gr.Accordion("Advanced Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.1,
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
                                value=50,
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
        gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")

        msg.submit(
            fn=bot,
            inputs=[msg, chatbot, temperature, top_p, top_k, repetition_penalty],
            outputs=[msg, chatbot, streamer],
            queue=True,
        )
        submit.click(
            fn=bot,
            inputs=[msg, chatbot, temperature, top_p, top_k, repetition_penalty],
            outputs=[msg, chatbot, streamer],
            queue=True,
        )
        stop.click(fn=stop_chat, inputs=streamer, outputs=[streamer], queue=False)
        clear.click(fn=stop_chat_and_clear_history, inputs=streamer, outputs=[chatbot, streamer], queue=False)

        return demo