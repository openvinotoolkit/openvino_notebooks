from __future__ import annotations

import queue
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""


class OpenVINOLLM(LLM):
    """OpenVINO Pipeline API.

    To use, you should have the ``openvino-genai`` python package installed.

    Example using from_model_path:
        .. code-block:: python

            from langchain_community.llms import OpenVINOLLM
            ov = OpenVINOPipeline.from_model_path(
                model_path="./openvino_model_dir",
                device="CPU",
            )
    Example passing pipeline in directly:
        .. code-block:: python

            import openvino_genai
            pipe = openvino_genai.LLMPipeline("./openvino_model_dir", "CPU")
            config = openvino_genai.GenerationConfig()
            ov = OpenVINOPipeline.from_model_path(
                pipe=pipe,
                config=config,
            )

    """

    pipe: Any = None
    tokenizer: Any = None
    config: Any = None
    streamer: Any = None

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        device: str = "CPU",
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> OpenVINOLLM:
        """Construct the oepnvino object from model_path"""
        try:
            import openvino_genai

        except ImportError:
            raise ImportError("Could not import OpenVINO GenAI package. " "Please install it with `pip install openvino-genai`.")

        class IterableStreamer(openvino_genai.StreamerBase):
            """
            A custom streamer class for handling token streaming
            and detokenization with buffering.

            Attributes:
                tokenizer (Tokenizer): The tokenizer used for encoding
                and decoding tokens.
                tokens_cache (list): A buffer to accumulate tokens
                for detokenization.
                text_queue (Queue): A synchronized queue
                for storing decoded text chunks.
                print_len (int): The length of the printed text
                to manage incremental decoding.
            """

            def __init__(self, tokenizer: Any) -> None:
                """
                Initializes the IterableStreamer with the given tokenizer.

                Args:
                    tokenizer (Tokenizer): The tokenizer to use for encoding
                    and decoding tokens.
                """
                super().__init__()
                self.tokenizer = tokenizer
                self.tokens_cache: list[int] = []
                self.text_queue: Any = queue.Queue()
                self.print_len = 0

            def __iter__(self) -> IterableStreamer:
                """
                Returns the iterator object itself.
                """
                return self

            def __next__(self) -> str:
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

            def get_stop_flag(self) -> bool:
                """
                Checks whether the generation process should be stopped.

                Returns:
                    bool: Always returns False in this implementation.
                """
                return False

            def put_word(self, word: Any) -> None:
                """
                Puts a word into the text queue.

                Args:
                    word (str): The word to put into the queue.
                """
                self.text_queue.put(word)

            def put(self, token_id: int) -> bool:
                """
                Processes a token and manages the decoding buffer.
                Adds decoded text to the queue.

                Args:
                    token_id (int): The token_id to process.

                Returns:
                    bool: True if generation should be stopped, False otherwise.
                """
                self.tokens_cache.append(token_id)
                text = self.tokenizer.decode(self.tokens_cache, skip_special_tokens=True)

                word = ""
                if len(text) > self.print_len and "\n" == text[-1]:
                    word = text[self.print_len :]
                    self.tokens_cache = []
                    self.print_len = 0
                elif len(text) >= 3 and text[-3:] == chr(65533):
                    pass
                elif len(text) > self.print_len:
                    word = text[self.print_len :]
                    self.print_len = len(text)
                self.put_word(word)

                if self.get_stop_flag():
                    self.end()
                    return True
                else:
                    return False

            def end(self) -> None:
                """
                Flushes residual tokens from the buffer
                and puts a None value in the queue to signal the end.
                """
                text = self.tokenizer.decode(self.tokens_cache, skip_special_tokens=True)
                if len(text) > self.print_len:
                    word = text[self.print_len :]
                    self.put_word(word)
                    self.tokens_cache = []
                    self.print_len = 0
                self.put_word(None)

            def reset(self) -> None:
                """
                Resets the state.
                """
                self.tokens_cache = []
                self.text_queue = queue.Queue()
                self.print_len = 0

        class ChunkStreamer(IterableStreamer):
            def __init__(self, tokenizer: Any, tokens_len: int = 4) -> None:
                super().__init__(tokenizer)
                self.tokens_len = tokens_len

            def put(self, token_id: int) -> bool:
                if (len(self.tokens_cache) + 1) % self.tokens_len != 0:
                    self.tokens_cache.append(token_id)
                    return False
                return super().put(token_id)

        pipe = openvino_genai.LLMPipeline(model_path, device, **kwargs)

        config = pipe.get_generation_config()
        if tokenizer is None:
            tokenizer = pipe.get_tokenizer()
        streamer = ChunkStreamer(tokenizer)

        return cls(
            pipe=pipe,
            tokenizer=tokenizer,
            config=config,
            streamer=streamer,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to OpenVINO's generate request."""
        if stop is not None:
            self.config.stop_strings = set(stop)
        try:
            import openvino as ov
            import openvino_genai

        except ImportError:
            raise ImportError("Could not import OpenVINO GenAI package. " "Please install it with `pip install openvino-genai`.")
        if not isinstance(self.tokenizer, openvino_genai.Tokenizer):
            tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            prompt = openvino_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))
        output = self.pipe.generate(prompt, self.config, **kwargs)
        if not isinstance(self.tokenizer, openvino_genai.Tokenizer):
            output = self.tokenizer.batch_decode(output.tokens, skip_special_tokens=True)[0]
        return output

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Output OpenVINO's generation Stream"""
        from threading import Event, Thread

        if stop is not None:
            self.config.stop_strings = set(stop)
        try:
            import openvino as ov
            import openvino_genai

        except ImportError:
            raise ImportError("Could not import OpenVINO GenAI package. " "Please install it with `pip install openvino-genai`.")
        if not isinstance(self.tokenizer, openvino_genai.Tokenizer):
            tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            prompt = openvino_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))
        stream_complete = Event()

        def generate_and_signal_complete() -> None:
            """
            genration function for single thread
            """
            self.streamer.reset()
            self.pipe.generate(prompt, self.config, self.streamer, **kwargs)
            stream_complete.set()
            self.streamer.end()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        for char in self.streamer:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {}

    @property
    def _llm_type(self) -> str:
        return "openvino_pipeline"


DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""


class ChatOpenVINO(BaseChatModel):
    """OpenVINO LLM's as ChatModels.

    Works with `OpenVINOLLM` LLMs.

    See full list of supported init args and their descriptions in the params
    section.

    Instantiate:
        .. code-block:: python

            from langchain_community.llms import OpenVINOLLM
            llm = OpenVINOPipeline.from_model_path(
                model_path="./openvino_model_dir",
                device="CPU",
            )

            chat = ChatOpenVINO(llm=llm, verbose=True)

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user
                sentence to French."),
                ("human", "I love programming."),
            ]

            chat(...).invoke(messages)

        .. code-block:: python


    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk)

        .. code-block:: python


    """  # noqa: E501

    llm: Any
    """LLM, must be of type OpenVINOLLM"""
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if self.tokenizer is None:
            self.tokenizer = self.llm.tokenizer

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = self.llm._generate(prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs)
        return self._to_chat_result(llm_result)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = self._to_chat_prompt(messages)

        for data in self.llm.stream(request, **kwargs):
            delta = data
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                run_manager.on_llm_new_token(delta, chunk=chunk)
            yield chunk

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        try:
            import openvino_genai

        except ImportError:
            raise ImportError("Could not import OpenVINO GenAI package. " "Please install it with `pip install openvino-genai`.")
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage!")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return (
            self.tokenizer.apply_chat_template(messages_dicts, add_generation_prompt=True)
            if isinstance(self.tokenizer, openvino_genai.Tokenizer)
            else self.tokenizer.apply_chat_template(messages_dicts, tokenize=False, add_generation_prompt=True)
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(message=AIMessage(content=g.text), generation_info=g.generation_info)
            chat_generations.append(chat_generation)

        return ChatResult(generations=chat_generations, llm_output=llm_result.llm_output)

    @property
    def _llm_type(self) -> str:
        return "openvino-chat-wrapper"
