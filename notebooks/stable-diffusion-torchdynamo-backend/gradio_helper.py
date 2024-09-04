import numpy as np
import time
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


modes = {
    "txt2img": "Text to Image",
    "img2img": "Image to Image",
}

time_stamps = []


def callback(iter, t, latents):
    time_stamps.append(time.time())


def error_str(error, title="Error"):
    return (
        f"""#### {title}
            {error}"""
        if error
        else ""
    )


def on_mode_change(mode):
    return gr.update(visible=mode == modes["img2img"]), gr.update(visible=mode == modes["txt2img"])


# TODO Consider passing pipe instead of model_id
def make_demo(model_id):
    def inference(
        inf_mode,
        prompt,
        guidance=7.5,
        steps=25,
        width=768,
        height=768,
        seed=-1,
        img=None,
        strength=0.5,
        neg_prompt="",
    ):
        if seed == -1:
            seed = np.randint(0, 10000000)
        generator = torch.Generator().manual_seed(seed)
        res = None

        global time_stamps, pipe
        time_stamps = []
        try:
            if inf_mode == modes["txt2img"]:
                if type(pipe).__name__ != "StableDiffusionPipeline":
                    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
                    pipe.unet = torch.compile(pipe.unet, backend="openvino")
                res = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=int(steps),
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    generator=generator,
                    callback=callback,
                    callback_steps=1,
                ).images
            elif inf_mode == modes["img2img"]:
                if img is None:
                    return (
                        None,
                        None,
                        gr.update(
                            visible=True,
                            value=error_str("Image is required for Image to Image mode"),
                        ),
                    )
                if type(pipe).__name__ != "StableDiffusionImg2ImgPipeline":
                    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
                    pipe.unet = torch.compile(pipe.unet, backend="openvino")
                res = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    image=img,
                    num_inference_steps=int(steps),
                    strength=strength,
                    guidance_scale=guidance,
                    generator=generator,
                    callback=callback,
                    callback_steps=1,
                ).images
        except Exception as e:
            return None, None, gr.update(visible=True, value=error_str(e))

        warmup_duration = time_stamps[1] - time_stamps[0]
        generation_rate = (steps - 1) / (time_stamps[-1] - time_stamps[1])
        res_info = "Warm up time: " + str(round(warmup_duration, 2)) + " secs "
        if generation_rate >= 1.0:
            res_info = res_info + ", Performance: " + str(round(generation_rate, 2)) + " it/s "
        else:
            res_info = res_info + ", Performance: " + str(round(1 / generation_rate, 2)) + " s/it "

        return (
            res,
            gr.update(visible=True, value=res_info),
            gr.update(visible=False, value=None),
        )

    with gr.Blocks(css="style.css") as demo:
        gr.HTML(
            f"""
                Model used: {model_id}         
            """
        )
        with gr.Row():
            with gr.Column(scale=60):
                with gr.Group():
                    prompt = gr.Textbox(
                        "a photograph of an astronaut riding a horse",
                        label="Prompt",
                        max_lines=2,
                    )
                    neg_prompt = gr.Textbox(
                        "frames, borderline, text, character, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur",
                        label="Negative prompt",
                    )
                    res_img = gr.Gallery(label="Generated images", show_label=False)
                error_output = gr.Markdown(visible=False)

            with gr.Column(scale=40):
                generate = gr.Button(value="Generate")

                with gr.Group():
                    inf_mode = gr.Dropdown(list(modes.values()), label="Inference Mode", value=modes["txt2img"])

                    with gr.Column(visible=False) as i2i:
                        image = gr.Image(label="Image", height=128, type="pil")
                        strength = gr.Slider(
                            label="Transformation strength",
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.5,
                        )

                with gr.Group():
                    with gr.Row() as txt2i:
                        width = gr.Slider(label="Width", value=512, minimum=64, maximum=1024, step=8)
                        height = gr.Slider(label="Height", value=512, minimum=64, maximum=1024, step=8)

                with gr.Group():
                    with gr.Row():
                        steps = gr.Slider(label="Steps", value=20, minimum=1, maximum=50, step=1)
                        guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)

                    seed = gr.Slider(-1, 10000000, label="Seed (-1 = random)", value=-1, step=1)

                res_info = gr.Markdown(visible=False)

        inf_mode.change(on_mode_change, inputs=[inf_mode], outputs=[i2i, txt2i], queue=False)

        inputs = [
            inf_mode,
            prompt,
            guidance,
            steps,
            width,
            height,
            seed,
            image,
            strength,
            neg_prompt,
        ]

        outputs = [res_img, res_info, error_output]
        prompt.submit(inference, inputs=inputs, outputs=outputs)
        generate.click(inference, inputs=inputs, outputs=outputs)

    return demo
