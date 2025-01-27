import gradio as gr
import sys
from tqdm.auto import tqdm
import openvino as ov
import numpy as np
from PIL import Image, ImageFilter
import openvino_genai as ov_genai
from pathlib import Path
import requests

MAX_SEED = np.iinfo(np.int32).max


def image_to_tensor(image: Image) -> ov.Tensor:
    pic = image.convert("RGB")
    image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
    return ov.Tensor(image_data)


def prepare_examples():
    data = {
        "bench.jpg": "https://github.com/user-attachments/assets/509367b6-a252-4782-8be6-197152ec4203",
        "sofa.jpg": "https://github.com/user-attachments/assets/8345025a-56af-42d3-a222-a976421449cd",
        "mountain.jpg": "https://github.com/user-attachments/assets/163f3e5b-2ad9-4c71-82b6-8daa2ebcd74e",
        "sunset.jpg": "https://github.com/user-attachments/assets/0f63c661-bb05-4114-8573-facd83b58cac",
        "castle.jpg": "https://github.com/user-attachments/assets/80f0750d-8ec4-42e3-915c-e2cc8631830c",
        "dog.jpg": "https://github.com/user-attachments/assets/6b8dc877-28b3-42e0-9e7f-3ec383867cd5",
    }
    for file_name, url in data.items():
        if not Path(file_name).exists():
            Image.open(requests.get(url, stream=True).raw).resize((512, 512)).save(file_name)
    return list(data)


def make_demo(pipe):
    def predict(
        input_data,
        prompt="",
        negative_prompt="",
        guidance_scale=7.5,
        steps=20,
        strength=1.0,
        invert_mask=False,
        blur_mask=0,
        seed=0,
        randomize_seed=True,
        progress=gr.Progress(track_tqdm=True),
    ):
        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)
        generator = ov_genai.TorchGenerator(seed)
        init_image = input_data["background"]
        mask = Image.fromarray(np.array(input_data["layers"][-1])[:, :, -1] if not invert_mask else 255 - np.array(input_data["layers"][-1])[:, :, -1])
        if blur_mask > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur_mask))
        init_image_tensor = image_to_tensor(init_image)
        mask_tensor = image_to_tensor(mask)
        pbar = tqdm()

        def callback(step, num_steps, latent):
            if pbar.total is None:
                pbar.reset(num_steps)
            pbar.update(1)
            sys.stdout.flush()
            return False

        output = pipe.generate(
            prompt,
            init_image_tensor,
            mask_tensor,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=int(steps),
            strength=strength,
            generator=generator,
            callback=callback,
        )
        image = Image.fromarray(output.data[0])
        return image, seed

    examples = prepare_examples()
    with gr.Blocks() as demo:
        with gr.Column():
            image = gr.ImageMask(type="pil", height="100wv")
            image_out = gr.Image(interactive=False, height="100wv")
        prompt = gr.Textbox(placeholder="Your prompt (what you want in place of what is erased)", show_label=False)
        btn = gr.Button("Inpaint!")

        with gr.Accordion(label="Advanced Settings", open=False):
            guidance_scale = gr.Number(value=7.5, minimum=1.0, maximum=20.0, step=0.1, label="guidance_scale")
            steps = gr.Number(value=20, minimum=10, maximum=50, step=1, label="steps")
            strength = gr.Number(value=1.0, minimum=0.01, maximum=1.0, step=0.05, label="strength")
            negative_prompt = gr.Textbox(label="negative_prompt", placeholder="Your negative prompt", info="what you don't want to see in the image")
            invert_mask = gr.Checkbox(label="invert mask", value=False)
            blur_mask = gr.Slider(
                label="Blurring factor",
                minimum=0,
                maximum=100,
                step=1,
                value=0,
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

        btn.click(
            fn=predict,
            inputs=[image, prompt, negative_prompt, guidance_scale, steps, strength, invert_mask, blur_mask, seed, randomize_seed],
            outputs=[image_out, seed],
            api_name="run",
        )
        prompt.submit(
            fn=predict,
            inputs=[image, prompt, negative_prompt, guidance_scale, steps, strength, invert_mask, blur_mask, seed, randomize_seed],
            outputs=[image_out, seed],
        )
        gr.Examples(examples, inputs=image)

    return demo
