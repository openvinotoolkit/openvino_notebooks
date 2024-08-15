import gradio as gr
import numpy as np


def make_demo(pipeline, preprocess, postprocess, default_image_path):
    def generate_from_image(img, text, seed, num_steps, strength, _=gr.Progress(track_tqdm=True)):
        preprocessed_img, meta_data = preprocess(img)
        np.random.seed(seed)
        result = pipeline(text, preprocessed_img, num_inference_steps=num_steps, strength=strength)
        result_img = postprocess(result["images"][0], meta_data["src_width"], meta_data["src_height"])
        return result_img

    with gr.Blocks() as demo:
        with gr.Tab("Image-to-Image generation"):
            with gr.Row():
                with gr.Column():
                    i2i_input = gr.Image(label="Image", type="pil")
                    i2i_text_input = gr.Textbox(lines=3, label="Text")
                    i2i_seed_input = gr.Slider(0, 1024, value=42, step=1, label="Seed")
                    i2i_steps_input = gr.Slider(1, 50, value=10, step=1, label="Steps")
                    strength_input = gr.Slider(0, 1, value=0.5, label="Strength")
                i2i_out = gr.Image(label="Result")
            i2i_btn = gr.Button()
            sample_i2i_text = "amazing watercolor painting"
            i2i_btn.click(
                generate_from_image,
                [
                    i2i_input,
                    i2i_text_input,
                    i2i_seed_input,
                    i2i_steps_input,
                    strength_input,
                ],
                i2i_out,
            )
            gr.Examples(
                [[str(default_image_path), sample_i2i_text, 42, 10, 0.5]],
                [
                    i2i_input,
                    i2i_text_input,
                    i2i_seed_input,
                    i2i_steps_input,
                    strength_input,
                ],
            )
        return demo
