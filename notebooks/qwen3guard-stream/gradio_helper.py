from collections.abc import Callable, Iterator

import gradio as gr


def make_demo(
    moderate_prompt: Callable[[str], dict],
    moderate_stream: Callable[[str, str], Iterator[list]],
    generate: Callable[[str], tuple[str, list]],
) -> gr.Blocks:
    """Build three views sharing one serialized moderation session."""
    with gr.Blocks() as demo:
        gr.Markdown(
            "# Qwen3Guard-Stream with OpenVINO\n"
            "Model predictions are not a guarantee of safety. Unsafe is blocked during generation; "
            "Controversial is reported. The fixed-response tab visualizes predictions without blocking."
        )
        with gr.Tab("User prompt"):
            prompt = gr.Textbox(label="User prompt", value="How can I grow tomatoes on a balcony?", max_lines=6)
            check = gr.Button("Moderate")
            verdict = gr.JSON(label="Input moderation")
            check.click(moderate_prompt, inputs=prompt, outputs=verdict, concurrency_id="guard", concurrency_limit=1)
        with gr.Tab("Fixed-response stream"):
            stream_prompt = gr.Textbox(label="User prompt", value="How can I grow tomatoes on a balcony?")
            response = gr.Textbox(label="Assistant response", value="Choose a sunny spot and water the plants regularly.", lines=3)
            stream = gr.Button("Inspect token by token")
            table = gr.Dataframe(
                headers=["Token", "Risk", "Probability", "Category", "Flagged"],
                datatype=["str", "str", "number", "str", "bool"],
                interactive=False,
                label="Predictions for the accumulated conversation, including chat-template tokens",
            )
            stream.click(moderate_stream, inputs=[stream_prompt, response], outputs=table, concurrency_id="guard", concurrency_limit=1)
        with gr.Tab("Guarded generation"):
            generation_prompt = gr.Textbox(label="User prompt", value="Give two short tips for growing tomatoes.")
            run = gr.Button("Generate with moderation")
            answer = gr.Textbox(label="Retained assistant answer")
            report = gr.JSON(label="Prompt and response verdicts")
            run.click(generate, inputs=generation_prompt, outputs=[answer, report], concurrency_id="guard", concurrency_limit=1)
    return demo.queue(default_concurrency_limit=1)
