import gradio as gr
import torch
from diffusers.utils import export_to_video, load_image
import numpy as np
import requests
from PIL import Image
from io import BytesIO

MAX_SEED = np.iinfo(np.int32).max

# Use raw content URL for GitHub
raw_url = "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/i2v_input.JPG"
response = requests.get(raw_url)
img = Image.open(BytesIO(response.content))
img.save("i2v_input.jpg")


def make_demo(pipeline):
    def generate_video(
        prompt, negative_prompt, image, guidance_scale=1.0, seed=42, height=832, width=480, num_inference_steps=4, progress=gr.Progress(track_tqdm=True)
    ):
        image = load_image(image)
        output = pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=20,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
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
            gr.Image(label="Input Image", type="pil"),
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
            gr.Slider(
                label="Height",
                minimum=320,
                maximum=1024,
                step=32,
                value=832,
            ),
            gr.Slider(
                label="Width",
                minimum=320,
                maximum=1024,
                step=32,
                value=480,
            ),
            gr.Slider(
                label="Inference Steps",
                minimum=1,
                maximum=50,
                step=1,
                value=4,
            ),
        ],
        outputs=gr.Video(label="Generated Video"),
        title="Wan2.2-TI2V-5B OpenVINO Video Generator",
        flagging_mode="never",
        examples=[
            [
                "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
                "",
                "i2v_input.jpg",
                5.0,
                42,
                832,
                480,
                4,
            ],
        ],
    )
    return iface
