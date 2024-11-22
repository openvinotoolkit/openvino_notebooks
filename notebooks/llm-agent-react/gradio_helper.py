from typing import Callable
import gradio as gr


examples = [
    ["Based on current weather in London, show me a picture of Big Ben through its URL"],
    ["What is OpenVINO ?"],
    ["Create an image of pink cat and return its URL"],
    ["How many people live in Canada ?"],
    ["What is the weather like in New York now ?"],
]


def handle_user_message(message, history):
    """
    callback function for updating user messages in interface on submit button click

    Params:
      message: current message
      history: conversation history
    Returns:
      None
    """
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def make_demo(run_fn: Callable, stop_fn: Callable):
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        gr.Markdown(f"""<h1><center>AI Agent with OpenVINO and LangChain</center></h1>""")
        chatbot = gr.Chatbot(height=800)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                    container=False,
                )
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
        gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")

        submit_event = msg.submit(
            fn=handle_user_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=run_fn,
            inputs=[
                chatbot,
            ],
            outputs=chatbot,
            queue=True,
        )
        submit_click_event = submit.click(
            fn=handle_user_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=run_fn,
            inputs=[
                chatbot,
            ],
            outputs=chatbot,
            queue=True,
        )
        stop.click(
            fn=stop_fn,
            inputs=None,
            outputs=None,
            cancels=[submit_event, submit_click_event],
            queue=False,
        )
        clear.click(lambda: None, None, chatbot, queue=False)
    return demo
