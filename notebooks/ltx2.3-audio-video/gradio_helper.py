import uuid

import gradio as gr
import torch
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import encode_video


def make_demo(pipe, generation_mode):
    is_image_to_video = generation_mode == "image-to-video"

    def generate(prompt, negative_prompt, image, width, height, num_frames, num_inference_steps, seed):
        if is_image_to_video and image is None:
            raise gr.Error("Provide an input image for image-to-audio-video generation.")

        frame_rate = 24.0
        generation_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "num_frames": int(num_frames),
            "frame_rate": frame_rate,
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": 3.0,
            "generator": torch.Generator("cpu").manual_seed(int(seed)),
            "output_type": "np",
            "return_dict": False,
        }
        if is_image_to_video:
            generation_args.update({"image": image, "image_crf": 0})

        video, audio = pipe(**generation_args)
        output_path = f"ltx2.3_output_{uuid.uuid4().hex[:8]}.mp4"
        encode_video(
            video[0],
            fps=frame_rate,
            audio=audio[0].float().cpu(),
            audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
            output_path=output_path,
        )
        return output_path

    with gr.Blocks() as demo:
        gr.Markdown("# LTX-2.3 synchronized audio-video generation")
        prompt = gr.Textbox(
            label="Prompt",
            value="A cat stretches lazily on a sunny windowsill, purring softly, while birds chirp outside.",
            lines=4,
        )
        negative_prompt = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2)
        image = gr.Image(label="Input image", type="pil", visible=is_image_to_video)
        with gr.Row():
            width = gr.Slider(minimum=64, maximum=1280, value=768, step=32, label="Width")
            height = gr.Slider(minimum=64, maximum=720, value=512, step=32, label="Height")
            num_frames = gr.Slider(minimum=9, maximum=257, value=121, step=8, label="Number of frames")
        with gr.Row():
            num_inference_steps = gr.Slider(minimum=1, maximum=50, value=30, step=1, label="Inference steps")
            seed = gr.Slider(minimum=0, maximum=2**31 - 1, value=42, step=1, label="Seed")
        generate_button = gr.Button("Generate")
        result = gr.Video(label="Generated audio-video")

        generate_button.click(
            generate,
            inputs=[prompt, negative_prompt, image, width, height, num_frames, num_inference_steps, seed],
            outputs=result,
        )

    return demo
