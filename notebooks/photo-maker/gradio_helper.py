from typing import Callable
import gradio as gr


trigger_word = "img"
prompt = "sci-fi, closeup portrait photo of a man img in Iron man suit, face"
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"


def make_demo(fn: Callable):
    with gr.Blocks() as demo:
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(label="Your image", sources=["upload"], type="pil")
                output_image = gr.Image(label="Generated Images", type="pil")
            positive_input = gr.Textbox(label=f"Text prompt, Trigger words is '{trigger_word}'")
            neg_input = gr.Textbox(label="Negative prompt")
            with gr.Row():
                seed_input = gr.Slider(0, 10_000_000, value=42, label="Seed")
                steps_input = gr.Slider(label="Steps", value=10, minimum=5, maximum=50, step=1)
                style_strength_ratio_input = gr.Slider(label="Style strength ratio", value=20, minimum=5, maximum=100, step=5)
                btn = gr.Button()
            btn.click(
                fn=fn,
                inputs=[
                    positive_input,
                    input_image,
                    neg_input,
                    seed_input,
                    steps_input,
                    style_strength_ratio_input,
                ],
                outputs=output_image,
            )
            gr.Examples(
                [
                    [prompt, negative_prompt],
                    [
                        "A woman img wearing a Christmas hat",
                        negative_prompt,
                    ],
                    [
                        "A man img in a helmet and vest riding a motorcycle",
                        negative_prompt,
                    ],
                    [
                        "photo of a middle-aged man img sitting on a plush leather couch, and watching television show",
                        negative_prompt,
                    ],
                    [
                        "photo of a skilled doctor img in a pristine white lab coat enjoying a delicious meal in a sophisticated dining room",
                        negative_prompt,
                    ],
                    [
                        "photo of superman img flying through a vibrant sunset sky, with his cape billowing in the wind",
                        negative_prompt,
                    ],
                ],
                [positive_input, neg_input],
            )
    return demo
