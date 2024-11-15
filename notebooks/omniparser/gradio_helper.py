import gradio as gr

MARKDOWN = """
# OpenVINO OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

def make_demo(process_fn):

    with gr.Blocks() as demo:
        gr.Markdown(MARKDOWN)
        with gr.Row():
            with gr.Column():
                image_input_component = gr.Image(
                    type='filepath', label='Upload image')
                # set the threshold for removing the bounding boxes with low confidence, default is 0.05
                box_threshold_component = gr.Slider(
                    label='Box Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.05)
                # set the threshold for removing the bounding boxes with large overlap, default is 0.1
                iou_threshold_component = gr.Slider(
                    label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1)
                imgsz_component = gr.Slider(
                    label='Icon Detect Image Size', minimum=640, maximum=1920, step=32, value=640)
                submit_button_component = gr.Button(
                    value='Submit', variant='primary')
            with gr.Column():
                image_output_component = gr.Image(type='pil', label='Image Output')
                text_output_component = gr.Textbox(label='Parsed screen elements', placeholder='Text Output')

        submit_button_component.click(
            fn=process_fn,
            inputs=[
                image_input_component,
                box_threshold_component,
                iou_threshold_component,
                imgsz_component
            ],
            outputs=[image_output_component, text_output_component]
        )
    return demo