import gradio as gr
import numpy as np
from PIL import Image
import sys
from tqdm.auto import tqdm


def make_demo(pipeline):
    import openvino_genai as ov_genai

    def generate(prompt, negative_prompt, seed, num_steps, _=gr.Progress(track_tqdm=True)):
        pbar = tqdm(total=num_steps)

        def callback(step, num_steps, latent):
            if num_steps != pbar.total:
                pbar.reset(num_steps)
            pbar.update(1)
            sys.stdout.flush()
            return False

        result = pipeline.generate(
            prompt, negative_prompt=negative_prompt, num_inference_steps=num_steps, generator=ov_genai.TorchGenerator(seed), callback=callback
        )

        pbar.close()
        return Image.fromarray(result.data[0])

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


def make_demo_zoom_video(pipeline, generate_video):
    def generate(
        prompt,
        negative_prompt,
        seed,
        steps,
        frames,
        edge_size,
        zoom_in,
        progress=gr.Progress(track_tqdm=True),
    ):
        np.random.seed(seed)
        video_path = generate_video(
            pipeline,
            prompt,
            negative_prompt,
            num_inference_steps=steps,
            num_frames=frames,
            mask_width=edge_size,
            zoom_in=zoom_in,
        )
        np.random.seed(None)

        return video_path.replace(".mp4", ".gif")

    gr.close_all()
    demo = gr.Interface(
        generate,
        [
            gr.Textbox(
                "valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k",
                label="Prompt",
            ),
            gr.Textbox("lurry, bad art, blurred, text, watermark", label="Negative prompt"),
            gr.Slider(value=9999, label="Seed", step=1, maximum=10000000),
            gr.Slider(value=20, label="Steps", minimum=1, maximum=50),
            gr.Slider(value=3, label="Frames", minimum=1, maximum=50),
            gr.Slider(value=128, label="Edge size", minimum=32, maximum=256),
            gr.Checkbox(label="Zoom in"),
        ],
        "image",
    )
    return demo
