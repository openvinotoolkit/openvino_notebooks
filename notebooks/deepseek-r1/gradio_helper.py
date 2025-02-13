import openvino as ov
import openvino_genai as ov_genai
from uuid import uuid4
from threading import Event, Thread
import queue
import sys
import re

max_new_tokens = 256

core = ov.Core()

chinese_examples = [
    ["向 5 岁的孩子解释重力。"],
    ["给我讲一个关于微积分的笑话。"],
    ["编写代码时需要避免哪些常见错误？"],
    ["撰写一篇关于“人工智能和 OpenVINO 的优势”的 100 字博客文章"],
    ["求解方程：2x + 5 = 15"],
    ["说出养猫的 5 个优点"],
    ["简化 (-k + 4) + (-2 + 3k)"],
    ["求半径为20的圆的面积"],
    ["对未来5年AI趋势进行预测"],
]

english_examples = [
    ["Explain gravity to a 5-year-old."],
    ["Tell me a joke about calculus."],
    ["What are some common mistakes to avoid when writing code?"],
    ["Write a 100-word blog post on “Benefits of Artificial Intelligence and OpenVINO“"],
    ["Solve the equation: 2x + 5 = 15."],
    ["Name 5 advantages to be a cat"],
    ["Simplify (-k + 4) + (-2 + 3k)"],
    ["Find the area of ​​a circle with radius 20"],
    ["Make a forecast about AI trends for next 5 years"],
]


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

DEFAULT_SYSTEM_PROMPT_CHINESE = """\
你是一个乐于助人、尊重他人以及诚实可靠的助手。在安全的情况下，始终尽可能有帮助地回答。 您的回答不应包含任何有害、不道德、种族主义、性别歧视、有毒、危险或非法的内容。请确保您的回答在社会上是公正的和积极的。
如果一个问题没有任何意义或与事实不符，请解释原因，而不是回答错误的问题。如果您不知道问题的答案，请不要分享虚假信息。另外，答案请使用中文。\
"""


def get_system_prompt(model_language, system_prompt=None):
    if system_prompt is not None:
        return system_prompt
    return DEFAULT_SYSTEM_PROMPT_CHINESE if (model_language == "Chinese") else DEFAULT_SYSTEM_PROMPT


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
        if (len(self.tokens_cache) + 1) % self.tokens_len != 0:
            self.tokens_cache.append(token_id)
            return False
        sys.stdout.flush()
        return super().put(token_id)


def make_demo(pipe, model_configuration, model_id, model_language, disable_advanced=False):
    import gradio as gr

    start_message = get_system_prompt(model_language, model_configuration.get("system_prompt"))
    if "genai_chat_template" in model_configuration:
        pipe.get_tokenizer().set_chat_template(model_configuration["genai_chat_template"])

    def get_uuid():
        """
        universal unique identifier for thread
        """
        return str(uuid4())

    def default_partial_text_processor(partial_text: str, new_text: str):
        """
        helper for updating partially generated answer, used by default

        Params:
        partial_text: text buffer for storing previosly generated text
        new_text: text update for the current step
        Returns:
        updated text string

        """
        new_text = re.sub(r'^<think>', '<em><small>I am thinking...', new_text)
        new_text = re.sub('</think>', 'I think I know the answer</small></em>', new_text)
        partial_text += new_text
        return partial_text

    text_processor = model_configuration.get("partial_text_processor", default_partial_text_processor)

    def bot(message, history, temperature, top_p, top_k, repetition_penalty, max_tokens):
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
        streamer = ChunkStreamer(pipe.get_tokenizer())
        if not disable_advanced:
            config = pipe.get_generation_config()
            config.temperature = temperature
            config.top_p = top_p
            config.top_k = top_k
            config.do_sample = temperature > 0.0
            config.max_new_tokens = max_tokens
            config.repetition_penalty = repetition_penalty
            if "stop_strings" in model_configuration:
                config.stop_strings = set(model_configuration["stop_strings"])
        else:
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = max_tokens
        history = history or []
        if not history:
            pipe.start_chat(system_message=start_message)

        history.append([message, ""])
        new_prompt = message

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
        pipe.finish_chat()
        streamer.reset()
        return None, None

    examples = chinese_examples if (model_language == "Chinese") else english_examples

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
                    clear = gr.Button("Clear")
        with gr.Row(visible=not disable_advanced):
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
                    with gr.Column():
                        with gr.Row():
                            max_tokens = gr.Slider(
                                label="Max new tokens",
                                value=256,
                                minimum=128,
                                maximum=1024,
                                step=32,
                                interactive=True,
                                info=("Maximum new tokens added to answer. Higher value can work for long response, but require more time to complete"),
                            )
        gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")

        msg.submit(
            fn=bot,
            inputs=[msg, chatbot, temperature, top_p, top_k, repetition_penalty, max_tokens],
            outputs=[msg, chatbot, streamer],
            queue=True,
        )
        submit.click(
            fn=bot,
            inputs=[msg, chatbot, temperature, top_p, top_k, repetition_penalty, max_tokens],
            outputs=[msg, chatbot, streamer],
            queue=True,
        )
        clear.click(fn=stop_chat_and_clear_history, inputs=streamer, outputs=[chatbot, streamer], queue=False)

        return demo
