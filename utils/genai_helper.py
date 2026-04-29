import openvino_genai as ov_genai
import queue
import sys
from typing import Union


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
        self.decoded_lengths = []
        self._current_length = 0
        self.last_generated_length = 0
        self._stop_flag = False

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
        value = self.text_queue.get()
        if value is None:
            raise StopIteration
        return value

    def get_stop_flag(self):
        """
        Checks if the stop flag has been set.
        Returns:
            StreamingStatus: CANCEL if the stop flag is set, otherwise RUNNING.
        """
        if self._stop_flag:
            return ov_genai.StreamingStatus.CANCEL
        return ov_genai.StreamingStatus.RUNNING

    def write_word(self, word: str):
        """
        Adds a decoded word to the text queue.

        Args:
            word (str): The decoded word to add.
        """
        self.text_queue.put(word)

    def write(self, token: Union[int, list[int]]) -> ov_genai.StreamingStatus:
        """
        Processes a token and manages the decoding buffer. Adds decoded text to the queue.

        Args:
            token (Union[int, list[int]]): The token(s) to process.

        Returns:
            StreamingStatus: RUNNING to continue, CANCEL to stop generation.
        """
        if isinstance(token, list):
            self.tokens_cache += token
            self.decoded_lengths += [-2 for _ in range(len(token) - 1)]
            self._current_length += len(token)
        else:
            self.tokens_cache.append(token)
            self._current_length += 1

        text = self.tokenizer.decode(self.tokens_cache)
        self.decoded_lengths.append(len(text))

        word = ""
        delay_n_tokens = 3
        if len(text) > self.print_len and "\n" == text[-1]:
            word = text[self.print_len :]
            self.tokens_cache = []
            self.decoded_lengths = []
            self.print_len = 0
        elif len(text) > 0 and text[-1] == chr(65533):
            self.decoded_lengths[-1] = -1
        elif len(self.tokens_cache) >= delay_n_tokens:
            self._compute_decoded_length(len(self.decoded_lengths) - delay_n_tokens)
            print_until = self.decoded_lengths[-delay_n_tokens]
            if print_until != -1 and print_until > self.print_len:
                word = text[self.print_len : print_until]
                self.print_len = print_until
        self.write_word(word)
        sys.stdout.flush()

        stop_flag = self.get_stop_flag()
        if stop_flag != ov_genai.StreamingStatus.RUNNING:
            self.end()
        return stop_flag

    def _compute_decoded_length(self, cache_position: int):
        """
        Lazily compute decoded length for a position (needed when tokens arrive in batches).

        Args:
            cache_position (int): The position in the cache to compute the decoded length for.
        """
        if self.decoded_lengths[cache_position] != -2:
            return
        cache_for_position = self.tokens_cache[: cache_position + 1]
        text_for_position = self.tokenizer.decode(cache_for_position)
        if len(text_for_position) > 0 and text_for_position[-1] == chr(65533):
            self.decoded_lengths[cache_position] = -1
        else:
            self.decoded_lengths[cache_position] = len(text_for_position)

    def end(self):
        """
        Flushes residual tokens from the buffer and puts a None value in the queue to signal the end.
        """
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.print_len:
            word = text[self.print_len :]
            self.write_word(word)
            self.tokens_cache = []
            self.print_len = 0
        self.last_generated_length = self._current_length
        self._current_length = 0
        self.text_queue.put(None)
        self._stop_flag = True

    def reset(self):
        """
        Resets the streamer to its initial state, clearing all buffers and queues.
        """
        self.tokens_cache = []
        self.text_queue = queue.Queue()
        self.print_len = 0
        self.decoded_lengths = []
        self._current_length = 0
        self.last_generated_length = 0
        self._stop_flag = False


class ChunkStreamer(IterableStreamer):

    def __init__(self, tokenizer, tokens_len=2):
        """
        Initializes the ChunkStreamer with the given tokenizer and token length.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for encoding and decoding tokens.
            tokens_len (int): The number of tokens to accumulate before processing.
        """
        super().__init__(tokenizer)
        self.tokens_len = tokens_len

    def write(self, token: Union[int, list[int]]) -> ov_genai.StreamingStatus:
        """
        Processes a token and manages the decoding buffer in chunks. Adds decoded text to the queue.
        Args:
            token (Union[int, list[int]]): The token(s) to process.
        """
        if isinstance(token, list):
            return super().write(token)
        if (len(self.tokens_cache) + 1) % self.tokens_len != 0:
            self.tokens_cache.append(token)
            self.decoded_lengths.append(-2)
            self._current_length += 1
            return ov_genai.StreamingStatus.RUNNING
        return super().write(token)
