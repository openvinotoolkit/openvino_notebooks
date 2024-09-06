from typing import Callable
import gradio as gr

examples = [
    "Give me a recipe for pizza with pineapple",
    "Write me a tweet about the new OpenVINO release",
    "Explain the difference between CPU and GPU",
    "Give five ideas for a great weekend with family",
    "Do Androids dream of Electric sheep?",
    "Who is Dolly?",
    "Please give me advice on how to write resume?",
    "Name 3 advantages to being a cat",
    "Write instructions on how to become a good AI engineer",
    "Write a love letter to my best friend",
]


def reset_textbox(instruction: str, response: str, perf: str):
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


def make_demo(run_fn: Callable, title: str = "Question Answering with OpenVINO"):
    with gr.Blocks() as demo:
        gr.Markdown(
            f"# {title}.\n"
            "Provide instruction which describes a task below or select among predefined examples and model writes response that performs requested task."
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
            run_fn,
            [user_text, top_p, temperature, top_k, max_new_tokens, performance],
            [model_output, performance],
        )
        button_submit.click(
            run_fn,
            [user_text, top_p, temperature, top_k, max_new_tokens, performance],
            [model_output, performance],
        )
        button_clear.click(
            reset_textbox,
            [user_text, model_output, performance],
            [user_text, model_output, performance],
        )
    return demo
