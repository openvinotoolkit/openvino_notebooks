"""Gradio demo for the OpenVINO Bernini-R pipeline.

Exposes the six Bernini tasks (t2i / i2i / t2v / v2v / r2v / rv2v). Image tasks
(t2i / i2i) return an image; the video tasks return an mp4. The visual-condition
inputs (source image / reference images / source video) are shown only for the
tasks that use them.
"""

import numpy as np
import gradio as gr

from ov_bernini_helper import TASK_GUIDANCE, TASK_SYSTEM_PROMPT, TASK_INPUTS

MAX_SEED = np.iinfo(np.int32).max

IMAGE_TASKS = ("t2i", "i2i")

TASK_INFO = {
    "t2i": "Text → Image",
    "i2i": "Image edit (reference image → image)",
    "t2v": "Text → Video",
    "v2v": "Video edit (source video → video)",
    "r2v": "Reference image(s) → Video",
    "rv2v": "Reference image(s) + source video → Video",
}

DEFAULT_NEG = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers"
)


def make_demo(pipeline):
    def generate(task, prompt, neg_prompt, image, images, video,
                 num_frames, num_inference_steps, omega_TI, omega_I, omega_V,
                 seed, height, width, fps, progress=gr.Progress(track_tqdm=True)):
        is_image = task in IMAGE_TASKS
        out_path = "bernini_output.png" if is_image else "bernini_output.mp4"
        needed = TASK_INPUTS[task]
        kwargs = dict(
            prompt=prompt,
            neg_prompt=neg_prompt,
            guidance_mode=TASK_GUIDANCE[task],
            system_prompt=TASK_SYSTEM_PROMPT[task],
            num_frames=1 if is_image else int(num_frames),
            num_inference_steps=int(num_inference_steps),
            omega_TI=float(omega_TI),
            omega_I=float(omega_I),
            omega_V=float(omega_V),
            seed=int(seed),
            height=int(height),
            width=int(width),
            fps=int(fps),
            output_path=out_path,
        )
        if "image" in needed and image is not None:
            kwargs["image"] = image
        if "images" in needed and images:
            kwargs["images"] = [im[0] if isinstance(im, (list, tuple)) else im for im in images]
        if "video" in needed and video is not None:
            kwargs["video"] = video

        result_path = pipeline(**kwargs)
        if is_image:
            return gr.update(value=result_path, visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(value=result_path, visible=True)

    def on_task_change(task):
        return (
            gr.update(visible=task == "i2i"),                       # image input
            gr.update(visible=task in ("r2v", "rv2v")),             # reference gallery
            gr.update(visible=task in ("v2v", "rv2v")),             # source video
            gr.update(visible=task not in IMAGE_TASKS),             # num_frames
        )

    with gr.Blocks() as demo:
        gr.Markdown("# Bernini-R-1.3B · OpenVINO\nUnified image / video generation & editing.")
        with gr.Row():
            with gr.Column():
                task = gr.Radio(
                    choices=list(TASK_INFO.keys()), value="t2i", label="Task",
                    info="t2i / i2i return an image; the rest return a video.",
                )
                gr.Markdown("\n".join(f"- **{k}** — {v}" for k, v in TASK_INFO.items()))
                prompt = gr.Textbox(label="Prompt", lines=3,
                                    value="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")
                image_in = gr.Image(label="Source image (i2i)", type="pil", visible=False)
                gallery_in = gr.Gallery(label="Reference image(s) (r2v / rv2v)", visible=False, type="pil")
                video_in = gr.Video(label="Source video (v2v / rv2v)", visible=False)
                with gr.Accordion("Advanced settings", open=False):
                    neg_prompt = gr.Textbox(label="Negative prompt", lines=2, value=DEFAULT_NEG)
                    num_inference_steps = gr.Slider(1, 60, value=40, step=1, label="Inference steps")
                    num_frames = gr.Slider(5, 81, value=20, step=4, label="Frames (video)", visible=False)
                    omega_TI = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="omega_TI (text guidance)")
                    omega_I = gr.Slider(1.0, 10.0, value=3.0, step=0.1, label="omega_I (image guidance)")
                    omega_V = gr.Slider(1.0, 10.0, value=3.0, step=0.1, label="omega_V (video guidance)")
                    with gr.Row():
                        height = gr.Slider(256, 1024, value=480, step=16, label="Height")
                        width = gr.Slider(256, 1024, value=832, step=16, label="Width")
                    with gr.Row():
                        seed = gr.Number(value=42, precision=0, label="Seed")
                        fps = gr.Slider(4, 24, value=16, step=1, label="FPS (video)")
                run = gr.Button("Generate", variant="primary")
            with gr.Column():
                out_image = gr.Image(label="Generated image", visible=True, interactive=False)
                out_video = gr.Video(label="Generated video", visible=False)

        task.change(on_task_change, inputs=task,
                    outputs=[image_in, gallery_in, video_in, num_frames])
        run.click(
            generate,
            inputs=[task, prompt, neg_prompt, image_in, gallery_in, video_in,
                    num_frames, num_inference_steps, omega_TI, omega_I, omega_V,
                    seed, height, width, fps],
            outputs=[out_image, out_video],
        )
    return demo
