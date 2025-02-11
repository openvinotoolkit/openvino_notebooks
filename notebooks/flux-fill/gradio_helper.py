import gradio as gr
from PIL import Image, ImageDraw


def can_expand(source_width, source_height, target_width, target_height, alignment):
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True


def prepare_image_and_mask(
    image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom
):
    target_size = (width, height)

    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    source = image.resize((new_width, new_height), Image.LANCZOS)

    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "75%":
        resize_percentage = 75
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:  # Custom
        resize_percentage = custom_resize_percentage

    # Calculate new dimensions based on percentage
    resize_factor = resize_percentage / 100
    new_width = int(source.width * resize_factor)
    new_height = int(source.height * resize_factor)

    # Ensure minimum size of 64 pixels
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)

    # Resize the image
    source = source.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the overlap in pixels based on the percentage
    overlap_x = int(new_width * (overlap_percentage / 100))
    overlap_y = int(new_height * (overlap_percentage / 100))

    # Ensure minimum overlap of 1 pixel
    overlap_x = max(overlap_x, 1)
    overlap_y = max(overlap_y, 1)

    # Calculate margins based on alignment
    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height

    # Adjust margins to eliminate gaps
    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    # Create a new background image and paste the resized source image
    background = Image.new("RGB", target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    # Create the mask
    mask = Image.new("L", target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Calculate overlap areas
    white_gaps_patch = 2

    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch

    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height

    # Draw the mask
    mask_draw.rectangle([(left_overlap, top_overlap), (right_overlap, bottom_overlap)], fill=0)

    return background, mask


def make_demo(pipe):
    def inpaint(
        image,
        width,
        height,
        overlap_percentage,
        num_inference_steps,
        resize_option,
        custom_resize_percentage,
        prompt_input,
        alignment,
        overlap_left,
        overlap_right,
        overlap_top,
        overlap_bottom,
        progress=gr.Progress(track_tqdm=True),
    ):

        background, mask = prepare_image_and_mask(
            image,
            width,
            height,
            overlap_percentage,
            resize_option,
            custom_resize_percentage,
            alignment,
            overlap_left,
            overlap_right,
            overlap_top,
            overlap_bottom,
        )

        if not can_expand(background.width, background.height, width, height, alignment):
            alignment = "Middle"

        cnet_image = background.copy()
        cnet_image.paste(0, (0, 0), mask)

        final_prompt = prompt_input

        # generator = torch.Generator(device="cuda").manual_seed(42)

        result = pipe(
            prompt=final_prompt,
            height=height,
            width=width,
            image=cnet_image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=30,
        ).images[0]

        result = result.convert("RGBA")
        cnet_image.paste(result, (0, 0), mask)

        return cnet_image, background

    def preview_image_and_mask(
        image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom
    ):
        background, mask = prepare_image_and_mask(
            image,
            width,
            height,
            overlap_percentage,
            resize_option,
            custom_resize_percentage,
            alignment,
            overlap_left,
            overlap_right,
            overlap_top,
            overlap_bottom,
        )

        preview = background.copy().convert("RGBA")
        red_overlay = Image.new("RGBA", background.size, (255, 0, 0, 64))
        red_mask = Image.new("RGBA", background.size, (0, 0, 0, 0))
        red_mask.paste(red_overlay, (0, 0), mask)
        preview = Image.alpha_composite(preview, red_mask)

        return preview

    def clear_result():
        return gr.update(value=None)

    def preload_presets(target_ratio, ui_width, ui_height):
        if target_ratio == "9:16":
            return 720, 1280, gr.update()
        elif target_ratio == "16:9":
            return 1280, 720, gr.update()
        elif target_ratio == "1:1":
            return 1024, 1024, gr.update()
        elif target_ratio == "Custom":
            return ui_width, ui_height, gr.update(open=True)

    def select_the_right_preset(user_width, user_height):
        if user_width == 720 and user_height == 1280:
            return "9:16"
        elif user_width == 1280 and user_height == 720:
            return "16:9"
        elif user_width == 1024 and user_height == 1024:
            return "1:1"
        else:
            return "Custom"

    def toggle_custom_resize_slider(resize_option):
        return gr.update(visible=(resize_option == "Custom"))

    def update_history(new_image, history):
        if history is None:
            history = []
        history.insert(0, new_image)
        return history

    css = """
    .gradio-container {
        max-width: 1250px !important;
    }
    """

    title = """<h1 align="center">FLUX Fill Outpaint</h1>
    <div align="center">Drop an image you would like to extend, pick your expected ratio and hit Generate.</div>
    <div align="center">Using <a href="https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev" target="_blank"><code>FLUX.1-Fill-dev</code></a></div>
    """

    with gr.Blocks(css=css) as demo:
        with gr.Column():
            gr.HTML(title)

            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Input Image")

                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt_input = gr.Textbox(label="Prompt (Optional)")
                        with gr.Column(scale=1):
                            run_button = gr.Button("Generate")

                    with gr.Row():
                        target_ratio = gr.Radio(label="Image Ratio", choices=["9:16", "16:9", "1:1", "Custom"], value="9:16", scale=3)
                        alignment_dropdown = gr.Dropdown(
                            choices=["Middle", "Left", "Right", "Top", "Bottom"],
                            value="Middle",
                            label="Alignment",
                        )
                    resize_option = gr.Radio(label="Resize input image", choices=["Full", "75%", "50%", "33%", "25%", "Custom"], value="75%")
                    custom_resize_percentage = gr.Slider(label="Custom resize (%)", minimum=1, maximum=100, step=1, value=50, visible=False)
                    with gr.Accordion(label="Advanced settings", open=False) as settings_panel:
                        with gr.Column():
                            with gr.Row():
                                width_slider = gr.Slider(
                                    label="Target Width",
                                    minimum=720,
                                    maximum=1536,
                                    step=8,
                                    value=720,
                                )
                                height_slider = gr.Slider(
                                    label="Target Height",
                                    minimum=720,
                                    maximum=1536,
                                    step=8,
                                    value=1280,
                                )

                            num_inference_steps = gr.Slider(label="Steps", minimum=2, maximum=50, step=1, value=28)
                            with gr.Group():
                                overlap_percentage = gr.Slider(label="Mask overlap (%)", minimum=1, maximum=50, value=10, step=1)
                                with gr.Row():
                                    overlap_top = gr.Checkbox(label="Overlap Top", value=True)
                                    overlap_right = gr.Checkbox(label="Overlap Right", value=True)
                                with gr.Row():
                                    overlap_left = gr.Checkbox(label="Overlap Left", value=True)
                                    overlap_bottom = gr.Checkbox(label="Overlap Bottom", value=True)

                            with gr.Column():
                                preview_button = gr.Button("Preview alignment and mask")

                with gr.Column():
                    result = gr.Image(
                        interactive=False,
                        label="Generated Image",
                    )
                    use_as_input_button = gr.Button("Use as Input Image", visible=False)
                    with gr.Accordion("History and Mask", open=False):
                        history_gallery = gr.Gallery(label="History", columns=6, object_fit="contain", interactive=False)
                        preview_image = gr.Image(label="Mask preview")

        def use_output_as_input(output_image):
            return output_image

        use_as_input_button.click(fn=use_output_as_input, inputs=[result], outputs=[input_image])

        target_ratio.change(
            fn=preload_presets, inputs=[target_ratio, width_slider, height_slider], outputs=[width_slider, height_slider, settings_panel], queue=False
        )

        width_slider.change(fn=select_the_right_preset, inputs=[width_slider, height_slider], outputs=[target_ratio], queue=False)

        height_slider.change(fn=select_the_right_preset, inputs=[width_slider, height_slider], outputs=[target_ratio], queue=False)

        resize_option.change(fn=toggle_custom_resize_slider, inputs=[resize_option], outputs=[custom_resize_percentage], queue=False)

        run_button.click(
            fn=clear_result,
            inputs=None,
            outputs=result,
        ).then(
            fn=inpaint,
            inputs=[
                input_image,
                width_slider,
                height_slider,
                overlap_percentage,
                num_inference_steps,
                resize_option,
                custom_resize_percentage,
                prompt_input,
                alignment_dropdown,
                overlap_left,
                overlap_right,
                overlap_top,
                overlap_bottom,
            ],
            outputs=[result, preview_image],
        ).then(
            fn=lambda x, history: update_history(x, history),
            inputs=[result, history_gallery],
            outputs=history_gallery,
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=use_as_input_button,
        )

        prompt_input.submit(
            fn=clear_result,
            inputs=None,
            outputs=result,
        ).then(
            fn=inpaint,
            inputs=[
                input_image,
                width_slider,
                height_slider,
                overlap_percentage,
                num_inference_steps,
                resize_option,
                custom_resize_percentage,
                prompt_input,
                alignment_dropdown,
                overlap_left,
                overlap_right,
                overlap_top,
                overlap_bottom,
            ],
            outputs=[result, preview_image],
        ).then(
            fn=lambda x, history: update_history(x, history),
            inputs=[result, history_gallery],
            outputs=history_gallery,
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=use_as_input_button,
        )

        preview_button.click(
            fn=preview_image_and_mask,
            inputs=[
                input_image,
                width_slider,
                height_slider,
                overlap_percentage,
                resize_option,
                custom_resize_percentage,
                alignment_dropdown,
                overlap_left,
                overlap_right,
                overlap_top,
                overlap_bottom,
            ],
            outputs=preview_image,
            queue=False,
        )

    return demo
