import gradio as gr


def make_demo(pipeline):
    def generate(text, seed, steps):
        return pipeline.text_to_image(text, num_steps=steps, seed=seed)[0]

    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(lines=3, label="Text"),
            gr.Slider(0, 10000000, value=45, label="Seed"),
            gr.Slider(1, 50, value=25, step=1, label="Steps"),
        ],
        outputs=gr.Image(label="Result"),
        examples=[
            ["photograph of an astronaut riding a horse", 80, 25],
            ["photograph of a cat", 45, 25],
        ],
        allow_flagging="never",
    )
    return demo
