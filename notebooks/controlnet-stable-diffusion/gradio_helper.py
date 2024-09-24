from pathlib import Path
from typing import Callable
import gradio as gr
import requests
import numpy as np

example_url = "https://user-images.githubusercontent.com/29454499/224540208-c172c92a-9714-4a7b-857a-b1e54b4d4791.jpg"

img_path = Path("example.jpg")

if not img_path.exists():
    r = requests.get(example_url)
    with img_path.open("wb") as f:
        f.write(r.content)


def make_demo(pipeline: Callable, pose_estimator: Callable):
    def generate(
        pose,
        prompt,
        negative_prompt,
        seed,
        num_steps,
        progress=gr.Progress(track_tqdm=True),
    ):
        np.random.seed(seed)
        result = pipeline(prompt, pose, num_steps, negative_prompt)[0]
        return result

    gr.close_all()
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                inp_img = gr.Image(label="Input image")
                pose_btn = gr.Button("Extract pose")
                examples = gr.Examples(["example.jpg"], inp_img)
            with gr.Column(visible=False) as step1:
                out_pose = gr.Image(label="Estimated pose", type="pil")
                inp_prompt = gr.Textbox("Dancing Darth Vader, best quality, extremely detailed", label="Prompt")
                inp_neg_prompt = gr.Textbox(
                    "monochrome, lowres, bad anatomy, worst quality, low quality",
                    label="Negative prompt",
                )
                inp_seed = gr.Slider(label="Seed", value=42, maximum=1024000000)
                inp_steps = gr.Slider(label="Steps", value=20, minimum=1, maximum=50)
                btn = gr.Button()
            with gr.Column(visible=False) as step2:
                out_result = gr.Image(label="Result")

        def extract_pose(img):
            if img is None:
                raise gr.Error("Please upload the image or use one from the examples list")
            return {
                step1: gr.update(visible=True),
                step2: gr.update(visible=True),
                out_pose: pose_estimator(img),
            }

        pose_btn.click(extract_pose, inp_img, [out_pose, step1, step2])
        btn.click(
            generate,
            [out_pose, inp_prompt, inp_neg_prompt, inp_seed, inp_steps],
            out_result,
        )
    return demo
