from typing import Callable
import gradio as gr
import numpy as np

examples = [
    [
        "Moore-AnimateAnyone/configs/inference/ref_images/anyone-2.png",
        "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4",
    ],
    [
        "Moore-AnimateAnyone/configs/inference/ref_images/anyone-10.png",
        "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-1_kps.mp4",
    ],
    [
        "Moore-AnimateAnyone/configs/inference/ref_images/anyone-11.png",
        "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-1_kps.mp4",
    ],
    [
        "Moore-AnimateAnyone/configs/inference/ref_images/anyone-3.png",
        "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4",
    ],
    [
        "Moore-AnimateAnyone/configs/inference/ref_images/anyone-5.png",
        "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4",
    ],
]


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Image(label="Reference Image", type="pil"),
            gr.Video(label="Pose video"),
            gr.Slider(
                label="Seed",
                value=42,
                minimum=np.iinfo(np.int32).min,
                maximum=np.iinfo(np.int32).max,
            ),
            gr.Slider(label="Guidance scale", value=3.5, minimum=1.1, maximum=10),
            gr.Slider(label="Number of inference steps", value=30, minimum=15, maximum=100),
        ],
        outputs="video",
        examples=examples,
        allow_flagging="never",
    )
    return demo
