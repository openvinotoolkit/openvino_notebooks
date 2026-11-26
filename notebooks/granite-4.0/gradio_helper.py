from collections.abc import Callable

import gradio as gr


def make_demo(generate: Callable[[list[dict[str, object]]], str]) -> gr.ChatInterface:
    def respond(message: str, history: list[dict[str, object]]) -> str:
        messages = [*history, {"role": "user", "content": message}]
        try:
            answer = generate(messages)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        # The model may legitimately end its turn immediately, e.g. after a short ambiguous request
        return answer if answer.strip() else "The model ended its answer without text. Please make the request more specific."

    return gr.ChatInterface(
        fn=respond,
        title="Granite 4.0-1B with OpenVINO",
        description="Chat locally with a bounded conversation. Clear the chat to start again.",
        examples=["What is OpenVINO?", "Explain weight compression in two sentences."],
        cache_examples=False,
        concurrency_limit=1,
    )
