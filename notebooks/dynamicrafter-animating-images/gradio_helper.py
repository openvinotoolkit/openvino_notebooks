from typing import Callable
import gradio as gr

css = """#input_img {max-width: 256px !important} #output_vid {max-width: 256px; max-height: 256px}"""

i2v_examples_256 = [
    ["DynamiCrafter/prompts/256/art.png", "man fishing in a boat at sunset", 50, 7.5, 1.0, 3, 234],
    ["DynamiCrafter/prompts/256/boy.png", "boy walking on the street", 50, 7.5, 1.0, 3, 125],
    ["DynamiCrafter/prompts/256/dance1.jpeg", "two people dancing", 50, 7.5, 1.0, 3, 116],
    ["DynamiCrafter/prompts/256/fire_and_beach.jpg", "a campfire on the beach and the ocean waves in the background", 50, 7.5, 1.0, 3, 111],
    ["DynamiCrafter/prompts/256/guitar0.jpeg", "bear playing guitar happily, snowing", 50, 7.5, 1.0, 3, 122],
]


def make_demo(fn: Callable):
    with gr.Blocks(analytics_enabled=False, css=css) as demo:
        with gr.Tab(label="Image2Video_256x256"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            i2v_input_image = gr.Image(label="Input Image", elem_id="input_img")
                        with gr.Row():
                            i2v_input_text = gr.Text(label="Prompts")
                        with gr.Row():
                            i2v_seed = gr.Slider(label="Random Seed", minimum=0, maximum=10000, step=1, value=123)
                            i2v_eta = gr.Slider(label="ETA", minimum=0.0, maximum=1.0, step=0.1, value=1.0, elem_id="i2v_eta")
                            i2v_cfg_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=15.0, step=0.5, value=7.5, elem_id="i2v_cfg_scale")
                        with gr.Row():
                            i2v_steps = gr.Slider(label="Sampling steps", minimum=1, maximum=60, step=1, value=50, elem_id="i2v_steps")
                            i2v_motion = gr.Slider(label="Motion magnitude", minimum=1, maximum=4, step=1, value=3, elem_id="i2v_motion")
                        i2v_end_btn = gr.Button("Generate")
                    with gr.Row():
                        i2v_output_video = gr.Video(label="Generated Video", elem_id="output_vid", autoplay=True, show_share_button=True)

                gr.Examples(
                    examples=i2v_examples_256,
                    inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed],
                    outputs=[i2v_output_video],
                    fn=fn,
                    cache_examples=False,
                )
            i2v_end_btn.click(
                fn=fn,
                inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed],
                outputs=[i2v_output_video],
            )
    return demo
