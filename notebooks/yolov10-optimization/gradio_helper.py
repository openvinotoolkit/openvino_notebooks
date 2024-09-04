from typing import Callable
import gradio as gr


def make_demo(fn: Callable, quantized: bool):
    with gr.Blocks() as demo:
        gr.HTML(
            """
        <h1 style='text-align: center'>
        YOLOv10: Real-Time End-to-End Object Detection using OpenVINO
        </h1>
        """
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", label="Image")
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.2,
                )
                iou_threshold = gr.Slider(
                    label="IoU Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.45,
                )
                use_int8 = gr.Checkbox(
                    value=quantized,
                    visible=quantized,
                    label="Use INT8 model",
                )
                yolov10_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Annotated Image")

            yolov10_infer.click(
                fn=fn,
                inputs=[
                    image,
                    use_int8,
                    conf_threshold,
                    iou_threshold,
                ],
                outputs=[output_image],
            )
        gr.Examples(["data/coco_bike.jpg"], inputs=[image])
    return demo
