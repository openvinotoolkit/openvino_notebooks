import gradio as gr
import numpy as np
from tqdm.auto import tqdm
import openvino as ov
import openvino_genai as ov_genai
import sys
from PIL import Image

sample_text = (
    "cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting, epic composition. "
    "A golden daylight, hyper-realistic environment. "
    "Hyper and intricate detail, photo-realistic. "
    "Cinematic and volumetric light. "
    "Epic concept art. "
    "Octane render and Unreal Engine, trending on artstation"
)


def image_to_tensor(image) -> ov.Tensor:
    pic = image.convert("RGB")
    image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
    return ov.Tensor(image_data)


def make_demo(pipeline):
    def generate_from_text(text, seed, num_steps, _=gr.Progress(track_tqdm=True)):
        random_generator = ov_genai.TorchGenerator(seed)

        pbar = tqdm(total=num_steps)

        def callback(step, num_steps, latent):
            if num_steps != pbar.total:
                pbar.reset(num_steps)
            pbar.update(1)
            sys.stdout.flush()
            return False

        result = pipeline.generate(text, num_inference_steps=num_steps, generator=random_generator, callback=callback)

        pbar.close()
        return Image.fromarray(result.data[0])

    with gr.Blocks() as demo:
        with gr.Tab("Text-to-Image generation"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(lines=3, label="Text")
                    seed_input = gr.Slider(0, 10000000, value=42, step=1, label="Seed")
                    steps_input = gr.Slider(1, 50, value=20, step=1, label="Steps")
                out = gr.Image(label="Result", type="pil")
            btn = gr.Button()
            btn.click(generate_from_text, [text_input, seed_input, steps_input], out)
            gr.Examples([[sample_text, 42, 20]], [text_input, seed_input, steps_input])
    return demo


def make_demo_i2i(pipeline, default_image_path):
    def generate_from_image(img, text, seed, num_steps, strength, _=gr.Progress(track_tqdm=True)):
        img_tensor = image_to_tensor(img)

        pbar = tqdm(total=int(num_steps * strength) + 1)

        def callback(step, num_steps, latent):
            if num_steps != pbar.total:
                pbar.reset(num_steps)
            pbar.update(1)
            sys.stdout.flush()
            return False

        random_generator = ov_genai.TorchGenerator(seed)
        result = pipeline.generate(text, img_tensor, num_inference_steps=num_steps, strength=strength, generator=random_generator, callaback=callback)
        pbar.close()
        return Image.fromarray(result.data[0])

    with gr.Blocks() as demo:
        with gr.Tab("Image-to-Image generation"):
            with gr.Row():
                with gr.Column():
                    i2i_input = gr.Image(label="Image", type="pil")
                    i2i_text_input = gr.Textbox(lines=3, label="Text")
                    i2i_seed_input = gr.Slider(0, 1024, value=42, step=1, label="Seed")
                    i2i_steps_input = gr.Slider(1, 50, value=10, step=1, label="Steps")
                    strength_input = gr.Slider(0, 1, value=0.25, label="Strength")
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
                [[str(default_image_path), sample_i2i_text, 42, 10, 0.25]],
                [
                    i2i_input,
                    i2i_text_input,
                    i2i_seed_input,
                    i2i_steps_input,
                    strength_input,
                ],
            )
        return demo
