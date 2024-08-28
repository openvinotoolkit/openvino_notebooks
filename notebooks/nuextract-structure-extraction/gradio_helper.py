import json
import gradio as gr
from typing import Callable

example_text = """We introduce Mistral 7B, a 7-billion-parameter language model engineered for
superior performance and efficiency. Mistral 7B outperforms the best open 13B
model (Llama 2) across all evaluated benchmarks, and the best released 34B
model (Llama 1) in reasoning, mathematics, and code generation. Our model
leverages grouped-query attention (GQA) for faster inference, coupled with sliding
window attention (SWA) to effectively handle sequences of arbitrary length with a
reduced inference cost. We also provide a model fine-tuned to follow instructions,
Mistral 7B - Instruct, that surpasses Llama 2 13B - chat model both on human and
automated benchmarks. Our models are released under the Apache 2.0 license.
Code: https://github.com/mistralai/mistral-src
Webpage: https://mistral.ai/news/announcing-mistral-7b/"""

example_schema = """{
    "Model": {
        "Name": "",
        "Number of parameters": "",
        "Number of max token": "",
        "Architecture": []
    },
    "Usage": {
        "Use case": [],
        "Licence": ""
    }
}"""


def handle_errors(fn: Callable):
    def wrapped_fn(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except json.JSONDecodeError as e:
            raise gr.Error(f"Invalid JSON Schema: {e}", duration=None)
        except Exception as e:
            raise gr.Error(e, duration=None)

    return wrapped_fn


def make_demo(fn: Callable):
    with gr.Blocks() as demo:
        gr.Markdown("# Structure Extraction with NuExtract and OpenVINO")

        with gr.Row():
            with gr.Column():
                text_textbox = gr.Textbox(
                    label="Text",
                    placeholder="Text from which to extract information",
                    lines=5,
                )
                schema_textbox = gr.Code(
                    label="JSON Schema",
                    language="json",
                    lines=5,
                )
            with gr.Column():
                model_output_textbox = gr.Code(
                    label="Model Response",
                    language="json",
                    interactive=False,
                    lines=10,
                )
        with gr.Row():
            gr.ClearButton(components=[text_textbox, schema_textbox, model_output_textbox])
            submit_button = gr.Button(value="Submit", variant="primary")
        with gr.Row():
            gr.Examples(examples=[[example_text, example_schema]], inputs=[text_textbox, schema_textbox])

        submit_button.click(
            fn=handle_errors(fn),
            inputs=[text_textbox, schema_textbox],
            outputs=[model_output_textbox],
        )
    return demo
