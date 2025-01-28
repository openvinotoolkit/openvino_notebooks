from typing import Callable
import gradio as gr

examples = [
    ["Tell me more about gaussian function"],
    ["Explain the formula of gaussian function to me"],
    ["What is the Herschel Maxwell derivation of a Gaussian ?"],
]


def clear_files():
    return "Vector Store is Not ready"


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


def make_demo(
    example_path: str,
    build_index: Callable,
    search: Callable,
    run_fn: Callable,
    stop_fn: Callable,
):

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        gr.Markdown("""<h1><center>QA over Video</center></h1>""")
        gr.Markdown(f"""<center>Powered by OpenVINO</center>""")
        image_list = gr.State([])
        txt_list = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                video_file = gr.Video(
                    label="Step 1: Load a '.mp4' video file",
                    value=example_path,
                )
                load_video = gr.Button("Step 2: Build Vector Store", variant="primary")
                status = gr.Textbox(
                    "Vector Store is Ready",
                    show_label=False,
                    max_lines=1,
                    interactive=False,
                )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=800,
                    label="Step 3: Input Query",
                )
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            msg = gr.Textbox(
                                label="QA Message Box",
                                placeholder="Chat Message Box",
                                show_label=False,
                                container=False,
                            )
                    with gr.Column():
                        with gr.Row():
                            submit = gr.Button("Submit", variant="primary")
                            stop = gr.Button("Stop")
                            clear = gr.Button("Clear")
                gr.Examples(
                    examples,
                    inputs=msg,
                    label="Click on any example and press the 'Submit' button",
                )
        video_file.clear(clear_files, outputs=[status], queue=False).then(lambda: gr.Button(interactive=False), outputs=submit)
        load_video.click(lambda: gr.Button(interactive=False), outputs=submit).then(
            fn=build_index,
            inputs=[video_file],
            outputs=[status],
            queue=True,
        ).then(lambda: gr.Button(interactive=True), outputs=submit)
        submit_event = (
            msg.submit(handle_user_message, [msg, chatbot], [msg, chatbot], queue=False)
            .then(
                search,
                [chatbot],
                [image_list, txt_list],
                queue=True,
            )
            .then(
                run_fn,
                [chatbot, image_list, txt_list],
                chatbot,
                queue=True,
            )
        )
        submit_click_event = (
            submit.click(handle_user_message, [msg, chatbot], [msg, chatbot], queue=False)
            .then(
                search,
                [chatbot],
                [image_list, txt_list],
                queue=True,
            )
            .then(
                run_fn,
                [chatbot, image_list, txt_list],
                chatbot,
                queue=True,
            )
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
