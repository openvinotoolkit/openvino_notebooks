import gradio as gr
import torch
from diffusers.utils import export_to_video
import numpy as np

MAX_SEED = np.iinfo(np.int32).max


def make_demo(pipeline):
    def generate_video(prompt, negative_prompt="", guidance_scale=1.0, seed=42, progress=gr.Progress(track_tqdm=True)):
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=20,
            guidance_scale=guidance_scale,
            num_inference_steps=4,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]

        video_path = "output.mp4"
        export_to_video(output, video_path, fps=10)
        return video_path

    iface = gr.Interface(
        fn=generate_video,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Enter your video prompt here"),
            gr.Textbox(label="Negative Prompt", placeholder="Optional negative prompt", value=""),
            gr.Slider(
                label="Guidance scale",
                minimum=0.0,
                maximum=20.0,
                step=0.1,
                value=1.0,
            ),
            gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=42,
            ),
        ],
        outputs=gr.Video(label="Generated Video"),
        title="Wan2.1-T2V-1.3B OpenVINO Video Generator",
        flagging_mode="never",
        examples=[
            ["a penguin playfully dancing in the snow, Antarctica", "", 1.0, 42],
            [
                "A cat walks on the grass, realistic",
                "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                2.5,
                678,
            ],
        ],
    )
    return iface
