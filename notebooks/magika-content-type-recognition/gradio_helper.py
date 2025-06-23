import gradio as gr


def make_demo(ov_magika):
    def classify(file_path):
        """Classify file using classes listing.
        Args:
            file_path): path to input file
        Returns:
            (dict): Mapping between class labels and class probabilities.
        """
        results = ov_magika.identify_bytes_topk(file_path)

        return {result.output.label: float(result.score) for result in results}

    demo = gr.Interface(
        fn=classify,
        inputs=[
            gr.File(label="Input file", type="binary"),
        ],
        outputs=gr.Label(label="Result"),
        examples=[["./README.md"]],
        allow_flagging="never",
    )
    return demo
