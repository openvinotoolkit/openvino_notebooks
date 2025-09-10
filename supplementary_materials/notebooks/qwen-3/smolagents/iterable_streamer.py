import queue
from typing import Union

import openvino_genai

class IterableStreamer(openvino_genai.StreamerBase):
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
        # get() will be blocked until a token is available.
        value = self.text_queue.get()
        if value is None:
            raise StopIteration
        return value

    def get_stop_flag(self):
        """
        Checks whether the generation process should be stopped or cancelled.

        Returns:
            openvino_genai.StreamingStatus: Always returns RUNNING in this implementation.
        """
        return openvino_genai.StreamingStatus.RUNNING

    def write_word(self, word: str):
        """
        Puts a word into the text queue.

        Args:
            word (str): The word to put into the queue.
        """
        self.text_queue.put(word)

    def write(self, token: Union[int, list[int]]) -> openvino_genai.StreamingStatus:
        """
        Processes a token and manages the decoding buffer. Adds decoded text to the queue.

        Args:
            token (Union[int, list[int]]): The token(s) to process.

        Returns:
            bool: True if generation should be stopped, False otherwise.
        """
        if type(token) is list:
            self.tokens_cache += token
            self.decoded_lengths += [-2 for _ in range(len(token) - 1)]
            self._current_length += len(token)
        else:
            self.tokens_cache.append(token)
            self._current_length += 1

        text = self.tokenizer.decode(self.tokens_cache)
        self.decoded_lengths.append(len(text))

        word = ""
        delay_n_tokens = 1
        if len(text) > self.print_len and "\n" == text[-1]:
            # Flush the cache after the new line symbol.
            word = text[self.print_len :]
            self.tokens_cache = []
            self.decoded_lengths = []
            self.print_len = 0
        elif len(text) > 0 and text[-1] == chr(65533):
            # Don't print incomplete text.
            self.decoded_lengths[-1] = -1
        elif len(self.tokens_cache) >= delay_n_tokens:
            self.compute_decoded_length_for_position(
                len(self.decoded_lengths) - delay_n_tokens
            )
            print_until = self.decoded_lengths[-delay_n_tokens]
            if print_until != -1 and print_until > self.print_len:
                # It is possible to have a shorter text after adding new token.
                # Print to output only if text length is increased and text is complete (print_until != -1).
                word = text[self.print_len : print_until]
                self.print_len = print_until
        self.write_word(word)

        stop_flag = self.get_stop_flag()
        if stop_flag != openvino_genai.StreamingStatus.RUNNING:
            # When generation is stopped from streamer then end is not called, need to call it here manually.
            self.end()

        return stop_flag

    def compute_decoded_length_for_position(self, cache_position: int):
        # decode was performed for this position, skippping
        if self.decoded_lengths[cache_position] != -2:
            return

        cache_for_position = self.tokens_cache[: cache_position + 1]
        text_for_position = self.tokenizer.decode(cache_for_position)

        if len(text_for_position) > 0 and text_for_position[-1] == chr(65533):
            # Mark text as incomplete
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