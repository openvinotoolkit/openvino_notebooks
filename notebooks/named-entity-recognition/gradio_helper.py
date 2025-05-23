import gradio as gr


def make_demo(ner_pipeline):

    def run_ner(text):
        output = ner_pipeline(text)
        return {"text": text, "entities": output}

    demo = gr.Interface(
        fn=run_ner,
        inputs=gr.Textbox(placeholder="Enter sentence here...", label="Input Text"),
        outputs=gr.HighlightedText(label="Output Text"),
        examples=[
            "My name is Wolfgang and I live in Berlin.",
        ],
        allow_flagging="never",
    )
    return demo
