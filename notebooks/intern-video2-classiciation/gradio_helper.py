import gradio as gr


def make_demo(classify):
    demo = gr.Interface(
        classify,
        [
            gr.Video(label="Video"),
            gr.Textbox(label="Labels", info="Comma-separated list of class labels"),
        ],
        gr.Label(label="Result"),
        examples=[["coco.mp4", "airplane, dog, car"]],
        allow_flagging="never",
    )

    return demo
