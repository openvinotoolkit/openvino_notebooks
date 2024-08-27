from pathlib import Path
from typing import Callable
import gradio as gr


def make_demo_llava(handle_user_message: Callable, run_chatbot: Callable, clear_history: Callable):
    title_markdown = """
    # 🌋 LLaVA: Large Language and Vision Assistant
    """

    tos_markdown = """
    ### Terms of use
    By using this service, users are required to agree to the following terms:
    The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
    """
    gr.close_all()
    with gr.Blocks(title="LLaVA") as demo:
        gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column():
                imagebox = gr.Image(type="pil")
                with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=512,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )

            with gr.Column(scale=3):
                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(height=400)
                    with gr.Row():
                        with gr.Column(scale=8):
                            textbox = gr.Textbox(
                                show_label=False,
                                placeholder="Enter text and press ENTER",
                                visible=True,
                                container=False,
                            )
                        with gr.Column(scale=1, min_width=60):
                            submit_btn = gr.Button(value="Submit", visible=True)
                    with gr.Row(visible=True) as button_row:
                        clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

        gr.Markdown(tos_markdown)

        submit_event = textbox.submit(
            fn=handle_user_message,
            inputs=[textbox, chatbot],
            outputs=[textbox, chatbot],
            queue=False,
        ).then(
            fn=run_chatbot,
            inputs=[imagebox, chatbot, temperature, top_p, max_output_tokens],
            outputs=chatbot,
            queue=True,
        )
        # Register listeners
        clear_btn.click(fn=clear_history, inputs=[textbox, imagebox, chatbot], outputs=[chatbot, textbox, imagebox])
        submit_click_event = submit_btn.click(
            fn=handle_user_message,
            inputs=[textbox, chatbot],
            outputs=[textbox, chatbot],
            queue=False,
        ).then(
            fn=run_chatbot,
            inputs=[imagebox, chatbot, temperature, top_p, max_output_tokens],
            outputs=chatbot,
            queue=True,
        )
    return demo


def make_demo_videollava(fn: Callable):
    examples_dir = Path("Video-LLaVA/videollava/serve/examples")
    gr.close_all()
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Image(label="Input Image", type="filepath"),
            gr.Video(label="Input Video"),
            gr.Textbox(label="Question"),
        ],
        outputs=gr.Textbox(lines=10),
        examples=[
            [
                f"{examples_dir}/extreme_ironing.jpg",
                None,
                "What is unusual about this image?",
            ],
            [
                f"{examples_dir}/waterview.jpg",
                None,
                "What are the things I should be cautious about when I visit here?",
            ],
            [
                f"{examples_dir}/desert.jpg",
                None,
                "If there are factual errors in the questions, point it out; if not, proceed answering the question. What’s happening in the desert?",
            ],
            [
                None,
                f"{examples_dir}/sample_demo_1.mp4",
                "Why is this video funny?",
            ],
            [
                None,
                f"{examples_dir}/sample_demo_3.mp4",
                "Can you identify any safety hazards in this video?",
            ],
            [
                None,
                f"{examples_dir}/sample_demo_9.mp4",
                "Describe the video.",
            ],
            [
                None,
                f"{examples_dir}/sample_demo_22.mp4",
                "Describe the activity in the video.",
            ],
            [
                f"{examples_dir}/sample_img_22.png",
                f"{examples_dir}/sample_demo_22.mp4",
                "Are the instruments in the pictures used in the video?",
            ],
            [
                f"{examples_dir}/sample_img_13.png",
                f"{examples_dir}/sample_demo_13.mp4",
                "Does the flag in the image appear in the video?",
            ],
            [
                f"{examples_dir}/sample_img_8.png",
                f"{examples_dir}/sample_demo_8.mp4",
                "Are the image and the video depicting the same place?",
            ],
        ],
        title="Video-LLaVA🚀",
        allow_flagging="never",
    )
    return demo
