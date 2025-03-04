import openvino_genai as ov_genai
import queue
import sys


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
        return self._stop_flag

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
        self.decoded_lengths.append(len(text))

        word = ""
        delay_n_tokens = 3
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
            print_until = self.decoded_lengths[-delay_n_tokens]
            if print_until != -1 and print_until > self.print_len:
                # It is possible to have a shorter text after adding new token.
                # Print to output only if text length is increased and text is complete (print_until != -1).
                word = text[self.print_len : print_until]
                self.print_len = print_until
        self.put_word(word)
        sys.stdout.flush()

        if self.get_stop_flag():
            # When generation is stopped from streamer then end is not called, need to call it here manually.
            self.end()
            return True  # True means stop generation
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
        self._stop_flag = True

    def reset(self):
        self.tokens_cache = []
        self.text_queue = queue.Queue()
        self.print_len = 0
        self.decoded_lengths = []
        self._stop_flag = False


class ChunkStreamer(IterableStreamer):

    def __init__(self, tokenizer, tokens_len=2):
        super().__init__(tokenizer)
        self.tokens_len = tokens_len

    def put(self, token_id: int) -> bool:
        if (len(self.tokens_cache) + 1) % self.tokens_len != 0:
            self.tokens_cache.append(token_id)
            self.decoded_lengths.append(-1)
            return False
        return super().put(token_id)
