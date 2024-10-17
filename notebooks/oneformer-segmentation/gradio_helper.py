from typing import Callable
import gradio as gr
import openvino as ov


def make_demo(run_fn: Callable, compile_model_fn: Callable, quantized: bool):
    available_devices = ov.Core().available_devices + ["AUTO"]
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                inp_img = gr.Image(label="Image", type="pil")
                inp_task = gr.Radio(["semantic", "instance", "panoptic"], label="Task", value="semantic")
                inp_device = gr.Dropdown(label="Device", choices=available_devices, value="AUTO")
            with gr.Column():
                out_result = gr.Image(label="Result (Original)" if quantized else "Result")
                inference_time = gr.Textbox(label="Time (seconds)")
                out_result_quantized = gr.Image(label="Result (Quantized)", visible=quantized)
                inference_time_quantized = gr.Textbox(label="Time (seconds)", visible=quantized)
        run_button = gr.Button(value="Run")
        run_button.click(
            fn=run_fn,
            inputs=[inp_img, inp_task, gr.Number(0, visible=False)],
            outputs=[out_result, inference_time],
        )
        run_quantized_button = gr.Button(value="Run quantized", visible=quantized)
        run_quantized_button.click(
            fn=run_fn,
            inputs=[inp_img, inp_task, gr.Number(1, visible=False)],
            outputs=[out_result_quantized, inference_time_quantized],
        )
        gr.Examples(examples=[["sample.jpg", "semantic"]], inputs=[inp_img, inp_task])

        def on_device_change_begin():
            return (
                run_button.update(value="Changing device...", interactive=False),
                run_quantized_button.update(value="Changing device...", interactive=False),
                inp_device.update(interactive=False),
            )

        def on_device_change_end():
            return (
                run_button.update(value="Run", interactive=True),
                run_quantized_button.update(value="Run quantized", interactive=True),
                inp_device.update(interactive=True),
            )

        inp_device.change(on_device_change_begin, outputs=[run_button, run_quantized_button, inp_device]).then(compile_model_fn, inp_device).then(
            on_device_change_end, outputs=[run_button, run_quantized_button, inp_device]
        )
    return demo
