import gradio as gr
import numpy as np
import torch
import random
from PIL import Image, ImageFilter

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048


def make_demo(pipe):
    def infer(
        edit_images,
        prompt,
        seed=42,
        randomize_seed=False,
        width=1024,
        height=1024,
        guidance_scale=3.5,
        control_strength=0.5,
        control_stop=0.33,
        num_inference_steps=50,
        progress=gr.Progress(track_tqdm=True),
    ):
        image = edit_images["background"].convert("RGB")
        mask = Image.fromarray(np.array(edit_images["layers"][-1])[:, :, -1])
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        out_image = pipe(
            prompt=prompt,
            inpaint_image=image,
            inpaint_mask=mask,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            control_strength=control_strength,
            control_stop=control_stop,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        return (image, out_image), seed

    css = """
:root {
    --primary-color: #7E57C2;
    --secondary-color: #5E35B1;
    --accent-color: #B39DDB;
    --background-color: #F5F5F7;
    --card-background: #FFFFFF;
    --text-color: #333333;
    --shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    --radius: 12px;
}
body {
    font-family: 'Inter', system-ui, sans-serif;
    background-color: var(--background-color);
}
#col-container {
    margin: 0 auto;
    max-width: 1200px;
    padding: 0;
}
.container {
    background-color: var(--card-background);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 24px;
    margin-bottom: 24px;
}
.header-container {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    border-radius: var(--radius);
    padding: 32px;
    margin-bottom: 24px;
    color: white;
    text-align: center;
    box-shadow: var(--shadow);
}
.header-container h1 {
    font-weight: 700;
    font-size: 2.5rem;
    margin-bottom: 8px;
    background: linear-gradient(to right, #ffffff, #e0e0e0);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-container p {
    font-size: 1.1rem;
    opacity: 0.92;
    margin-bottom: 16px;
}
.header-container a {
    color: var(--accent-color);
    text-decoration: underline;
    transition: opacity 0.2s;
}
.header-container a:hover {
    opacity: 0.8;
}
.btn-primary {
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    padding: 12px 24px;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(126, 87, 194, 0.3);
}
.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(126, 87, 194, 0.4);
}
.image-editor-container {
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
}
.prompt-container {
    background-color: var(--card-background);
    border-radius: var(--radius);
    padding: 16px;
    box-shadow: var(--shadow);
    margin-top: 16px;
}
.result-container {
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
}
.settings-container {
    background-color: var(--card-background);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow);
    margin-top: 16px;
}
.accordion-header {
    font-weight: 600;
    color: var(--primary-color);
}
/* Custom slider styling */
input[type="range"] {
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
}
input[type="range"]::-webkit-slider-thumb {
    background: var(--primary-color);
    border: 2px solid white;
    height: 18px;
    width: 18px;
}
.footer {
    text-align: center;
    padding: 24px;
    color: #777;
    font-size: 14px;
}
/* Animate the result transition */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
.result-animation {
    animation: fadeIn 0.5s ease-in-out;
}
    """

    with gr.Blocks(css=css, theme=gr.themes.Monochrome()) as demo:
        with gr.Column(elem_id="col-container"):
            # Header with gradient
            with gr.Column(elem_classes=["header-container"]):
                gr.HTML(
                    """
                    <h1>Flex.2 Preview Inpainting OpenVINO Demo</h1>
                """
                )
            # Main interface container
            with gr.Column(elem_classes=["container"]):
                with gr.Row():
                    # Left column: Input
                    with gr.Column(scale=1):
                        edit_image = gr.ImageEditor(
                            label="Upload and draw mask for inpainting",
                            type="pil",
                            sources=["upload", "webcam"],
                            image_mode="RGB",
                            layers=False,
                            brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                            height=500,
                        )

                        with gr.Column(elem_classes=["prompt-container"]):
                            prompt = gr.Text(
                                label="Your creative prompt",
                                show_label=True,
                                max_lines=1,
                                placeholder="Describe what you want to generate...",
                                container=True,
                            )

                            run_button = gr.Button("✨ Generate", elem_classes=["btn-primary"])
                    # Right column: Output
                    with gr.Column(scale=1, elem_classes=["result-container"]):
                        result = gr.ImageSlider(label="Before & After", type="pil", image_mode="RGB", elem_classes=["result-animation"])

            # Advanced settings in a nice container
            with gr.Column(elem_classes=["settings-container"]):
                with gr.Accordion("Advanced Settings", open=False, elem_classes=["accordion-header"]):
                    with gr.Column():
                        with gr.Row():
                            seed = gr.Slider(
                                label="Seed",
                                minimum=0,
                                maximum=MAX_SEED,
                                step=1,
                                value=0,
                            )
                            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                        with gr.Row():
                            height = gr.Slider(64, 2048, value=512, step=64, label="Height")
                            width = gr.Slider(64, 2048, value=512, step=64, label="Width")

                        with gr.Row():
                            guidance_scale = gr.Slider(0.0, 20.0, value=3.5, step=0.1, label="Guidance Scale")
                            control_strength = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Control Strength")

                        with gr.Row():
                            control_stop = gr.Slider(0.0, 1.0, value=0.33, step=0.05, label="Control Stop")
                            num_inference_steps = gr.Slider(1, 100, value=20, step=1, label="Inference Steps")

            # Footer
            gr.HTML(
                """
                <div class="footer">
                    <p>Flex.2 Preview Inpainting OpenVINO Demo</p>
                </div>
            """
            )

        run_button.click(
            fn=infer,
            inputs=[edit_image, prompt, seed, randomize_seed, width, height, guidance_scale, control_strength, control_stop, num_inference_steps],
            outputs=[result, seed],
        )

    return demo
