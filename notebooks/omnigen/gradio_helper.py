import gradio as gr
import random


def make_demo(pipe):
    def generate_image(
        text,
        img1,
        img2,
        img3,
        height,
        width,
        guidance_scale,
        img_guidance_scale,
        inference_steps,
        seed,
        max_input_image_size,
        randomize_seed,
        _=gr.Progress(track_tqdm=True),
    ):
        input_images = [img1, img2, img3]
        # Delete None
        input_images = [img for img in input_images if img is not None]
        if len(input_images) == 0:
            input_images = None

        if randomize_seed:
            seed = random.randint(0, 10000000)

        output = pipe(
            prompt=text,
            input_images=input_images,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            img_guidance_scale=img_guidance_scale,
            num_inference_steps=inference_steps,
            separate_cfg_infer=True,
            seed=seed,
            max_input_image_size=max_input_image_size,
        )
        img = output[0]
        return img

    def get_example():
        case = [
            [
                "A curly-haired man in a red shirt is drinking tea.",
                None,
                None,
                None,
                256,
                256,
                2.5,
                1.6,
                20,
                256,
            ],
            [
                "The woman in <img><|image_1|></img> waves her hand happily in the crowd",
                "OmniGen/imgs/test_cases/zhang.png",
                None,
                None,
                512,
                512,
                2.5,
                1.9,
                40,
                256,
            ],
            [
                "A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>.",
                "OmniGen/imgs/test_cases/two_man.jpg",
                None,
                None,
                256,
                256,
                2.5,
                1.6,
                20,
                256,
            ],
            [
                "The flower <img><|image_1|></img> is placed in the vase which is in the middle of <img><|image_2|></img> on a wooden table of a living room",
                "OmniGen/imgs/test_cases/rose.jpg",
                "OmniGen/imgs/test_cases/vase.jpg",
                None,
                512,
                512,
                2.5,
                1.6,
                20,
                512,
            ],
            [
                "<img><|image_1|><img>\n Remove the woman's earrings. Replace the mug with a clear glass filled with sparkling iced cola.",
                "OmniGen/imgs/demo_cases/t2i_woman_with_book.png",
                None,
                None,
                320,
                320,
                2.5,
                1.6,
                24,
                320,
            ],
            [
                "Detect the skeleton of human in this image: <img><|image_1|></img>.",
                "OmniGen/imgs/test_cases/control.jpg",
                None,
                None,
                512,
                512,
                2.0,
                1.6,
                20,
                512,
            ],
            [
                "Generate a new photo using the following picture and text as conditions: <img><|image_1|><img>\n A young boy is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him.",
                "OmniGen/imgs/demo_cases/skeletal.png",
                None,
                None,
                288,
                320,
                2,
                1.6,
                32,
                320,
            ],
            [
                "Following the depth mapping of this image <img><|image_1|><img>, generate a new photo: A young girl is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him.",
                "OmniGen/imgs/demo_cases/edit.png",
                None,
                None,
                512,
                512,
                2.0,
                1.6,
                15,
                512,
            ],
            [
                "<img><|image_1|><\/img> What item can be used to see the current time? Please highlight it in blue.",
                "OmniGen/imgs/test_cases/watch.jpg",
                None,
                None,
                224,
                224,
                2.5,
                1.6,
                100,
                224,
            ],
            [
                "According to the following examples, generate an output for the input.\nInput: <img><|image_1|></img>\nOutput: <img><|image_2|></img>\n\nInput: <img><|image_3|></img>\nOutput: ",
                "OmniGen/imgs/test_cases/icl1.jpg",
                "OmniGen/imgs/test_cases/icl2.jpg",
                "OmniGen/imgs/test_cases/icl3.jpg",
                224,
                224,
                2.5,
                1.6,
                12,
                768,
            ],
        ]
        return case

    description = """
    OmniGen is a unified image generation model that you can use to perform various tasks, including but not limited to text-to-image generation, subject-driven generation, Identity-Preserving Generation, and image-conditioned generation.
    For multi-modal to image generation, you should pass a string as `prompt`, and a list of image paths as `input_images`. The placeholder in the prompt should be in the format of `<img><|image_*|></img>` (for the first image, the placeholder is <img><|image_1|></img>. for the second image, the the placeholder is <img><|image_2|></img>).
    For example, use an image of a woman to generate a new image:
    prompt = "A woman holds a bouquet of flowers and faces the camera. Thw woman is \<img\>\<|image_1|\>\</img\>."
    Tips:
    - For image editing task and controlnet task, we recommend setting the height and width of output image as the same as input image. For example, if you want to edit a 512x512 image, you should set the height and width of output image as 512x512. You also can set the `use_input_image_size_as_output` to automatically set the height and width of output image as the same as input image.
    - If inference time is too long when inputting multiple images, please try to reduce the `max_input_image_size`. 
    - Oversaturated: If the image appears oversaturated, please reduce the `guidance_scale`.
    - Low-quality: More detailed prompts will lead to better results. 
    - Animate Style: If the generated images are in animate style, you can try to add `photo` to the prompt`.
    - Edit generated image. If you generate an image by omnigen and then want to edit it, you cannot use the same seed to edit this image. For example, use seed=0 to generate image, and should use seed=1 to edit this image.
    - For image editing tasks, we recommend placing the image before the editing instruction. For example, use `<img><|image_1|></img> remove suit`, rather than `remove suit <img><|image_1|></img>`. 
    """

    # Gradio
    with gr.Blocks() as demo:
        gr.Markdown("# OmniGen: Unified Image Generation")
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                # text prompt
                prompt_input = gr.Textbox(
                    label="Enter your prompt, use <img><|image_i|></img> to represent i-th input image", placeholder="Type your prompt here..."
                )

                with gr.Row(equal_height=True):
                    # input images
                    image_input_1 = gr.Image(label="<img><|image_1|></img>", type="filepath")
                    image_input_2 = gr.Image(label="<img><|image_2|></img>", type="filepath")
                    image_input_3 = gr.Image(label="<img><|image_3|></img>", type="filepath")

                # slider
                height_input = gr.Slider(label="Height", minimum=128, maximum=2048, value=256, step=16)
                width_input = gr.Slider(label="Width", minimum=128, maximum=2048, value=256, step=16)

                guidance_scale_input = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=5.0, value=2.5, step=0.1)

                img_guidance_scale_input = gr.Slider(label="img_guidance_scale", minimum=1.0, maximum=2.0, value=1.6, step=0.1)

                num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=20, step=1)

                seed_input = gr.Slider(label="Seed", minimum=0, maximum=2147483647, value=42, step=1)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                max_input_image_size = gr.Slider(label="max_input_image_size", minimum=128, maximum=2048, value=256, step=16)

                # generate
                generate_button = gr.Button("Generate Image")

            with gr.Column():
                # output image
                output_image = gr.Image(label="Output Image")

        # click
        generate_button.click(
            generate_image,
            inputs=[
                prompt_input,
                image_input_1,
                image_input_2,
                image_input_3,
                height_input,
                width_input,
                guidance_scale_input,
                img_guidance_scale_input,
                num_inference_steps,
                seed_input,
                max_input_image_size,
                randomize_seed,
            ],
            outputs=output_image,
        )

        gr.Examples(
            examples=get_example(),
            inputs=[
                prompt_input,
                image_input_1,
                image_input_2,
                image_input_3,
                height_input,
                width_input,
                guidance_scale_input,
                img_guidance_scale_input,
                seed_input,
                max_input_image_size,
                randomize_seed,
            ],
        )

    return demo
