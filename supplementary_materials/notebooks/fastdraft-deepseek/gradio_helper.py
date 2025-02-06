import openvino as ov
import openvino_genai as ov_genai
from uuid import uuid4
from threading import Event, Thread
import queue
import sys
import time


core = ov.Core()


english_examples = [
    ["Which is bigger, 9.9 or 9.11?"],
    ["Classify the following numbers as 'prime' or 'composite' - 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16.?"],
    ["What are the classifications of Academic Degrees?"],
    ["Which word does not belong to the other: Hammer, Screwdriver, Nail, Wood"],
    ["Identify which instrument is string or percussion: Kpanlogo, Shamisen"],
    ["Which of the following are colors: red, black, yellow, orange, sun, sunflower, chips, book, white, pink, blue, keyboard."],
]


class IterableStreamer(ov_genai.StreamerBase):
    """
    A custom streamer class for handling token streaming and detokenization with buffering.

    Attributes:
        tokenizer (Tokenizer): The tokenizer used for encoding and decoding tokens.
        tokens_cache (list): A buffer to accumulate tokens for detokenization.
        text_queue (Queue): A synchronized queue for storing decoded text chunks.
        print_len (int): The length of the printed text to manage incremental decoding.
    """

    def __init__(self, tokenizer):
        """
        Initializes the IterableStreamer with the given tokenizer.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for encoding and decoding tokens.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_cache = []
        self.text_queue = queue.Queue()
        self.print_len = 0

    def __iter__(self):
        """
        Returns the iterator object itself.
        """
        return self

    def __next__(self):
        """
        Returns the next value from the text queue.

        Returns:
            str: The next decoded text chunk.

        Raises:
            StopIteration: If there are no more elements in the queue.
        """
        value = self.text_queue.get()  # get() will be blocked until a token is available.
        if value is None:
            raise StopIteration
        return value

    def get_stop_flag(self):
        """
        Checks whether the generation process should be stopped.

        Returns:
            bool: Always returns False in this implementation.
        """
        return False

    def put_word(self, word: str):
        """
        Puts a word into the text queue.

        Args:
            word (str): The word to put into the queue.
        """
        self.text_queue.put(word)

    def put(self, token_id: int) -> bool:
        """
        Processes a token and manages the decoding buffer. Adds decoded text to the queue.

        Args:
            token_id (int): The token_id to process.

        Returns:
            bool: True if generation should be stopped, False otherwise.
        """
        self.tokens_cache.append(token_id)
        text = self.tokenizer.decode(self.tokens_cache)

        word = ""
        if len(text) > self.print_len and "\n" == text[-1]:
            # Flush the cache after the new line symbol.
            word = text[self.print_len :]
            self.tokens_cache = []
            self.print_len = 0
        elif len(text) >= 3 and text[-3:] == chr(65533):
            # Don't print incomplete text.
            pass
        elif len(text) > self.print_len:
            # It is possible to have a shorter text after adding new token.
            # Print to output only if text length is increaesed.
            word = text[self.print_len :]
            self.print_len = len(text)
        self.put_word(word)
        if self.get_stop_flag():
            # When generation is stopped from streamer then end is not called, need to call it here manually.
            self.end()
            return True  # True means stop  generation
        else:
            return False  # False means continue generation

    def end(self):
        """
        Flushes residual tokens from the buffer and puts a None value in the queue to signal the end.
        """
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.print_len:
            word = text[self.print_len :]
            self.put_word(word)
            self.tokens_cache = []
            self.print_len = 0
        self.put_word(None)

    def reset(self):
        self.tokens_cache = []
        self.text_queue = queue.Queue()
        self.print_len = 0


class ChunkStreamer(IterableStreamer):

    def __init__(self, tokenizer, tokens_len=4):
        super().__init__(tokenizer)
        self.tokens_len = tokens_len

    def put(self, token_id: int) -> bool:
        # print('check put', flush=True)
        if (len(self.tokens_cache) + 1) % self.tokens_len != 0:
            self.tokens_cache.append(token_id)
            return False
        sys.stdout.flush()
        return super().put(token_id)


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
