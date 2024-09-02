from pathlib import Path
from typing import Optional
from diffusers.utils import load_image, export_to_video
import PIL
import gradio as gr
import numpy as np
import torch

max_64_bit_int = 2**63 - 1

example_images_urls = [
    "https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/ship-7833921_1280.jpg?download=true",
    "https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/ai-generated-8476858_1280.png?download=true",
    "https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/ai-generated-8481641_1280.jpg?download=true",
    "https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/dog-7396912_1280.jpg?download=true",
    "https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/cupcakes-380178_1280.jpg?download=true",
]

example_images_dir = Path("example_images")
example_images_dir.mkdir(exist_ok=True)
example_imgs = []

for image_id, url in enumerate(example_images_urls):
    img = load_image(url)
    image_path = example_images_dir / f"{image_id}.png"
    img.save(image_path)
    example_imgs.append([image_path])


def resize_image(image, output_size=(512, 320)):
    # Calculate aspect ratios
    target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image

    # Resize then crop if the original image is larger
    if image_aspect > target_aspect:
        # Resize the image to match the target height, maintaining aspect ratio
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
        # Calculate coordinates for cropping
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        # Resize the image to match the target width, maintaining aspect ratio
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
        # Calculate coordinates for cropping
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image


def make_demo(pipeline):
    def sample(
        image: PIL.Image,
        seed: Optional[int] = 42,
        randomize_seed: bool = True,
        motion_bucket_id: int = 127,
        fps_id: int = 6,
        num_inference_steps: int = 15,
        num_frames: int = 4,
        max_guidance_scale=1.0,
        min_guidance_scale=1.0,
        decoding_t: int = 8,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
        output_folder: str = "outputs",
        progress=gr.Progress(track_tqdm=True),
    ):
        if image.mode == "RGBA":
            image = image.convert("RGB")

        if randomize_seed:
            seed = np.random.randint(0, max_64_bit_int)
        generator = torch.manual_seed(seed)

        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        base_count = len(list(output_folder.glob("*.mp4")))
        video_path = output_folder / f"{base_count:06d}.mp4"

        frames = pipeline(
            image,
            decode_chunk_size=decoding_t,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=0.1,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            max_guidance_scale=max_guidance_scale,
            min_guidance_scale=min_guidance_scale,
        ).frames[0]
        export_to_video(frames, str(video_path), fps=fps_id)

        return video_path, seed

    with gr.Blocks() as demo:
        gr.Markdown(
            """# Stable Video Diffusion: Image to Video Generation with OpenVINO.
    """
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="Upload your image", type="pil")
                generate_btn = gr.Button("Generate")
            video = gr.Video()
        with gr.Accordion("Advanced options", open=False):
            seed = gr.Slider(
                label="Seed",
                value=42,
                randomize=True,
                minimum=0,
                maximum=max_64_bit_int,
                step=1,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            motion_bucket_id = gr.Slider(
                label="Motion bucket id",
                info="Controls how much motion to add/remove from the image",
                value=127,
                minimum=1,
                maximum=255,
            )
            fps_id = gr.Slider(
                label="Frames per second",
                info="The length of your video in seconds will be num_frames / fps",
                value=6,
                minimum=5,
                maximum=30,
                step=1,
            )
            num_frames = gr.Slider(label="Number of Frames", value=8, minimum=2, maximum=25, step=1)
            num_steps = gr.Slider(label="Number of generation steps", value=4, minimum=1, maximum=8, step=1)
            max_guidance_scale = gr.Slider(
                label="Max guidance scale",
                info="classifier-free guidance strength",
                value=1.2,
                minimum=1,
                maximum=2,
            )
            min_guidance_scale = gr.Slider(
                label="Min guidance scale",
                info="classifier-free guidance strength",
                value=1,
                minimum=1,
                maximum=1.5,
            )
        examples = gr.Examples(
            examples=example_imgs,
            inputs=[image_in],
            outputs=[video, seed],
        )

        image_in.upload(fn=resize_image, inputs=image_in, outputs=image_in)
        generate_btn.click(
            fn=sample,
            inputs=[
                image_in,
                seed,
                randomize_seed,
                motion_bucket_id,
                fps_id,
                num_steps,
                num_frames,
                max_guidance_scale,
                min_guidance_scale,
            ],
            outputs=[video, seed],
            api_name="video",
        )

    return demo
