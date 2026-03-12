import gradio as gr


def make_demo(fn, title, description):
    with gr.Blocks() as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text with [MASK] to predict...",
                    lines=2,
                    info="Use the [MASK] token in your sentence to mask a word.",
                )
                submit_btn = gr.Button("Predict Mask", variant="primary")

                examples = gr.Examples(
                    examples=[
                        ["The capital of France is [MASK]."],
                        ["The quick brown fox jumps over the lazy [MASK]."],
                        ["I am going to the [MASK] to buy some milk."],
                        ["ModernBERT is a state-of-the-art model for natural language [MASK]."],
                        ["Please [MASK] the door when you leave."],
                        ["The weather today is [MASK] and sunny."],
                    ],
                    inputs=[input_text],
                )

            with gr.Column():
                output_label = gr.Label(num_top_classes=5, label="Top 5 Predictions")

        submit_btn.click(fn, inputs=[input_text], outputs=[output_label])

    return demo
