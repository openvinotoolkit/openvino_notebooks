import gradio as gr


def make_demo(pipeline):
    def generate(prompt, negative_prompt, seed, num_steps, _=gr.Progress(track_tqdm=True)):
        result = pipeline(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            seed=seed,
        )
        return result["sample"][0]

    gr.close_all()

    demo = gr.Interface(
        generate,
        [
            gr.Textbox(
                "valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k",
                label="Prompt",
            ),
            gr.Textbox(
                "frames, borderline, text, charachter, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur",
                label="Negative prompt",
            ),
            gr.Slider(value=42, label="Seed", maximum=10000000),
            gr.Slider(value=25, label="Steps", minimum=1, maximum=50),
        ],
        "image",
    )
    return demo
