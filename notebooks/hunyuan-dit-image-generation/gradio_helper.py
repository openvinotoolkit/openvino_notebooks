from typing import Callable
import gradio as gr
from hydit.constants import NEGATIVE_PROMPT

examples = [
    ["一只小猫"],
    ["a kitten"],
    ["一只聪明的狐狸走在阔叶树林里, 旁边是一条小溪, 细节真实, 摄影"],
    ["A clever fox walks in a broadleaf forest next to a stream, realistic details, photography"],
    ["请将“杞人忧天”的样子画出来"],
    ['Please draw a picture of "unfounded worries"'],
    ["枯藤老树昏鸦，小桥流水人家"],
    ["Withered vines, old trees and dim crows, small bridges and flowing water, people's houses"],
    ["湖水清澈，天空湛蓝，阳光灿烂。一只优雅的白天鹅在湖边游泳。它周围有几只小鸭子，看起来非常可爱，整个画面给人一种宁静祥和的感觉。"],
    [
        "The lake is clear, the sky is blue, and the sun is bright. An elegant white swan swims by the lake. There are several little ducks around it, which look very cute, and the whole picture gives people a sense of peace and tranquility."
    ],
    ["一朵鲜艳的红色玫瑰花，花瓣撒有一些水珠，晶莹剔透，特写镜头"],
    ["A bright red rose flower with petals sprinkled with some water drops, crystal clear, close-up"],
    ["风格是写实，画面主要描述一个亚洲戏曲艺术家正在表演，她穿着华丽的戏服，脸上戴着精致的面具，身姿优雅，背景是古色古香的舞台，镜头是近景"],
    [
        "The style is realistic. The picture mainly depicts an Asian opera artist performing. She is wearing a gorgeous costume and a delicate mask on her face. Her posture is elegant. The background is an antique stage and the camera is a close-up."
    ],
]


def make_demo(fn: Callable):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Input prompt", lines=3)
                with gr.Row():
                    infer_steps = gr.Slider(
                        label="Number Inference steps",
                        minimum=1,
                        maximum=200,
                        value=15,
                        step=1,
                    )
                    seed = gr.Number(
                        label="Seed",
                        minimum=-1,
                        maximum=1_000_000_000,
                        value=42,
                        step=1,
                        precision=0,
                    )
                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        negative_prompt = gr.Textbox(
                            label="Negative prompt",
                            value=NEGATIVE_PROMPT,
                            lines=2,
                        )
                    with gr.Row():
                        oriW = gr.Number(
                            label="Width",
                            minimum=768,
                            maximum=1024,
                            value=880,
                            step=16,
                            precision=0,
                            min_width=80,
                        )
                        oriH = gr.Number(
                            label="Height",
                            minimum=768,
                            maximum=1024,
                            value=880,
                            step=16,
                            precision=0,
                            min_width=80,
                        )
                        cfg_scale = gr.Slider(label="Guidance scale", minimum=1.0, maximum=16.0, value=7.5, step=0.5)
                with gr.Row():
                    advanced_button = gr.Button()
            with gr.Column():
                output_img = gr.Image(
                    label="Generated image",
                    interactive=False,
                )
            advanced_button.click(
                fn=fn,
                inputs=[
                    prompt,
                    negative_prompt,
                    seed,
                    infer_steps,
                    oriH,
                    oriW,
                ],
                outputs=output_img,
            )

        with gr.Row():
            gr.Examples(examples=examples, inputs=[prompt])
    return demo
