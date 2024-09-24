from typing import Callable
import gradio as gr
import openvino as ov

examples = [
    "Give me recipe for pizza with pineapple",
    "Write me a tweet about new OpenVINO release",
    "Explain difference between CPU and GPU",
    "Give five ideas for great weekend with family",
    "Do Androids dream of Electric sheep?",
    "Who is Dolly?",
    "Please give me advice how to write resume?",
    "Name 3 advantages to be a cat",
    "Write instructions on how to become a good AI engineer",
    "Write a love letter to my best friend",
]


def reset_textboxes(instruction: str, response: str, perf: str):
    """
    Helper function for resetting content of all text fields

    Parameters:
      instruction (str): Content of user instruction field.
      response (str): Content of model response field.
      perf (str): Content of performance info filed

    Returns:
      empty string for each placeholder
    """
    return "", "", ""


def make_demo(run_fn: Callable, select_device_fn: Callable):
    available_devices = ov.Core().available_devices + ["AUTO"]

    with gr.Blocks() as demo:
        gr.Markdown(
            "# Instruction following using Databricks Dolly 2.0 and OpenVINO.\n"
            "Provide insturction which describes a task below or select among predefined examples and model writes response that performs requested task."
        )

        with gr.Row():
            with gr.Column(scale=4):
                user_text = gr.Textbox(
                    placeholder="Write an email about an alpaca that likes flan",
                    label="User instruction",
                )
                model_output = gr.Textbox(label="Model response", interactive=False)
                performance = gr.Textbox(label="Performance", lines=1, interactive=False)
                with gr.Column(scale=1):
                    button_clear = gr.Button(value="Clear")
                    button_submit = gr.Button(value="Submit")
                gr.Examples(examples, user_text)
            with gr.Column(scale=1):
                device = gr.Dropdown(choices=available_devices, value="CPU", label="Device")
                max_new_tokens = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    value=256,
                    step=1,
                    interactive=True,
                    label="Max New Tokens",
                )
                top_p = gr.Slider(
                    minimum=0.05,
                    maximum=1.0,
                    value=0.92,
                    step=0.05,
                    interactive=True,
                    label="Top-p (nucleus sampling)",
                )
                top_k = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=0,
                    step=1,
                    interactive=True,
                    label="Top-k",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=5.0,
                    value=0.8,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )

        user_text.submit(
            fn=run_fn,
            inputs=[user_text, top_p, temperature, top_k, max_new_tokens, performance],
            outputs=[model_output, performance],
        )
        button_submit.click(fn=select_device_fn, inputs=[device, user_text], outputs=[user_text])
        button_submit.click(
            fn=run_fn,
            inputs=[user_text, top_p, temperature, top_k, max_new_tokens, performance],
            outputs=[model_output, performance],
        )
        button_clear.click(
            fn=reset_textboxes,
            inputs=[user_text, model_output, performance],
            outputs=[user_text, model_output, performance],
        )
        device.change(fn=select_device_fn, inputs=[device, user_text], outputs=[user_text])
    return demo
