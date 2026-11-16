import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def stream_model(monkeypatch):
    pytest.importorskip("langchain_core")
    pytest.importorskip("openvino")
    genai = pytest.importorskip("openvino_genai")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "utils"))
    monkeypatch.syspath_prepend(str(root / "notebooks" / "llm-rag-langchain"))
    from genai_helper import ChunkStreamer
    from ov_langchain_helper import OpenVINOLLM

    class BoundedWaitStreamer(ChunkStreamer):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            self.consumer_waiting = threading.Event()

        def __next__(self):
            # Capture the queue before allowing a delayed worker to run. A timeout
            # turns the old deadlock into a bounded failure without a leaked reader.
            text_queue = self.text_queue
            self.consumer_waiting.set()
            value = text_queue.get(timeout=2)
            if value is None:
                raise StopIteration
            return value

    tokenizer = Mock(spec=genai.Tokenizer)
    tokenizer.decode.side_effect = lambda tokens: "".join(chr(token) for token in tokens)
    streamer = BoundedWaitStreamer(tokenizer)
    pipe = Mock()
    llm = OpenVINOLLM(ov_pipe=pipe, tokenizer=tokenizer, config=SimpleNamespace(), streamer=streamer)
    return SimpleNamespace(llm=llm, pipe=pipe, streamer=streamer, tokenizer=tokenizer, genai=genai)


def test_stream_preserves_output_and_generation_options(stream_model):
    def generate(prompt, config, streamer, **kwargs):
        for token in map(ord, "hello world"):
            streamer.write(token)

    stream_model.pipe.generate.side_effect = generate
    manager = Mock()
    chunks = list(stream_model.llm._stream("question", stop=["STOP"], run_manager=manager, max_new_tokens=20))

    assert "".join(chunk.text for chunk in chunks) == "hello world"
    assert stream_model.llm.config.stop_strings == {"STOP"}
    stream_model.pipe.generate.assert_called_once_with("question", stream_model.llm.config, stream_model.streamer, max_new_tokens=20)
    assert [call.args[0] for call in manager.on_llm_new_token.call_args_list] == [chunk.text for chunk in chunks]


@pytest.mark.parametrize("partial_text", ["", "partial answer "])
def test_stream_reraises_generation_error_without_hanging(stream_model, partial_text):
    error = RuntimeError("device inference failed")

    def generate(prompt, config, streamer):
        if partial_text:
            streamer.write_word(partial_text)
        raise error

    stream_model.pipe.generate.side_effect = generate
    received = []
    with pytest.raises(RuntimeError, match="device inference failed") as caught:
        for text in stream_model.llm.stream("question"):
            received.append(text)

    assert caught.value is error
    assert "".join(received) == partial_text


def test_stream_resets_queue_before_starting_worker(stream_model, monkeypatch):
    workers = []
    original_thread = threading.Thread

    class DelayedWorker(original_thread):
        def run(self):
            if not stream_model.streamer.consumer_waiting.wait(timeout=2):
                raise RuntimeError("consumer did not start")
            super().run()

    def create_worker(*args, **kwargs):
        worker = DelayedWorker(*args, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(threading, "Thread", create_worker)
    stream_model.pipe.generate.side_effect = lambda prompt, config, streamer: streamer.write_word("answer")
    try:
        assert list(stream_model.llm.stream("question")) == ["answer"]
    finally:
        for worker in workers:
            worker.join(timeout=2)
            assert not worker.is_alive()


def test_stream_can_retry_after_generation_error(stream_model):
    stream_model.pipe.generate.side_effect = RuntimeError("first request failed")
    with pytest.raises(RuntimeError, match="first request failed"):
        list(stream_model.llm.stream("first"))

    stream_model.pipe.generate.side_effect = lambda prompt, config, streamer: streamer.write_word("recovered")
    assert list(stream_model.llm.stream("second")) == ["recovered"]


def test_stream_reraises_final_decode_error(stream_model):
    stream_model.tokenizer.decode.side_effect = RuntimeError("final decoding failed")
    stream_model.pipe.generate.return_value = None

    with pytest.raises(RuntimeError, match="final decoding failed"):
        list(stream_model.llm.stream("question"))


@pytest.mark.parametrize("consumer_exit", ["close", "callback_error"])
def test_stopping_consumer_cancels_and_joins_worker(stream_model, consumer_exit):
    cancelled = threading.Event()
    finished = threading.Event()
    release_worker = threading.Event()

    def generate(prompt, config, streamer):
        streamer.write_word("first")
        # Stay active until the consumer closes the iterator. The timeout bounds
        # failures on the old implementation and prevents background thread leaks.
        try:
            for _ in range(200):
                if release_worker.wait(timeout=0.01):
                    break
                if streamer.get_stop_flag() == stream_model.genai.StreamingStatus.CANCEL:
                    cancelled.set()
                    break
        finally:
            finished.set()

    stream_model.pipe.generate.side_effect = generate
    manager = Mock()
    if consumer_exit == "callback_error":
        manager.on_llm_new_token.side_effect = ValueError("callback failed")
    chunks = stream_model.llm._stream("question", run_manager=manager)
    try:
        if consumer_exit == "close":
            assert next(chunks).text == "first"
            chunks.close()
        else:
            with pytest.raises(ValueError, match="callback failed"):
                next(chunks)
        assert cancelled.is_set()
        assert finished.is_set()
    finally:
        release_worker.set()
        assert finished.wait(timeout=2)
