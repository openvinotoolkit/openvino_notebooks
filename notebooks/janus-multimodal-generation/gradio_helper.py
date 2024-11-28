import requests
from io import BytesIO
from pathlib import Path
from threading import Thread
import gradio as gr
import torch
from transformers import set_seed, TextIteratorStreamer
from PIL import Image
import numpy as np
from ov_janus_helper import generate_image


def download_example_images():
    image_urls = [
        "https://github.com/deepseek-ai/Janus/blob/main/images/pie_chart.png?raw=true",
        "https://github.com/deepseek-ai/Janus/blob/main/images/equation.png?raw=true",
    ]
    image_names = ["pie_chart.png", "equation.png"]

    for image_name, image_url in zip(image_names, image_urls):
        if not Path(image_name).exists():
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.save(image_name)


def make_demo(model, processor):
    download_example_images()

    # Multimodal Understanding function
    def multimodal_understanding(image, question, seed, top_p, temperature):
        # set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>\n{question}",
                "images": [image],
            },
            {"role": "Assistant", "content": ""},
        ]

        pil_images = [Image.fromarray(image)]
        prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True)

        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": prepare_inputs.attention_mask,
            "streamer": streamer,
            "max_new_tokens": 512,
            "pad_token_id": processor.tokenizer.eos_token_id,
            "bos_token_id": processor.tokenizer.bos_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
            "do_sample": False if temperature == 0 else True,
            "temperature": temperature,
            "top_p": top_p,
        }
        t = Thread(target=model.language_model.generate, kwargs=generate_kwargs)
        t.start()

        # Pull the generated text from the streamer, and update the model output.
        model_output = ""
        for new_text in streamer:
            model_output += new_text
            yield model_output
        return model_output

    def image_generation(prompt, seed, cfg_weight, num_images, progress=gr.Progress(track_tqdm=True)):
        set_seed(seed)
        images = generate_image(model, processor, prompt, cfg_weight=cfg_weight, parallel_size=int(num_images))
        images = [img.resize((1024, 1024), Image.LANCZOS) for img in images]
        return images

    # Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown(value="# Multimodal Understanding")
        # with gr.Row():
        with gr.Row():
            image_input = gr.Image()
            with gr.Column():
                question_input = gr.Textbox(label="Question")
                und_seed_input = gr.Number(label="Seed", precision=0, value=42)
                top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
                temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="temperature")

        understanding_button = gr.Button("Chat")
        understanding_output = gr.Textbox(label="Response")

        examples_vl = gr.Examples(
            label="Multimodal Understanding examples",
            examples=[
                [
                    "explain this chart",
                    "pie_chart.png",
                ],
                [
                    "Convert the formula into latex code.",
                    "equation.png",
                ],
            ],
            inputs=[question_input, image_input],
        )

        gr.Markdown(value="# Text-to-Image Generation")

        with gr.Row():
            cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight")

        prompt_input = gr.Textbox(label="Prompt")
        seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345)
        num_images = gr.Slider(minimum=1, maximum=32, step=1, value=2, label="Number of generated images")

        generation_button = gr.Button("Generate Images")

        image_output = gr.Gallery(label="Generated Images", columns=2, rows=2, height=300)

        examples_t2i = gr.Examples(
            label="Text to image generation examples. (Tips for designing prompts: Adding description like 'digital art' at the end of the prompt or writing the prompt in more detail can help produce better images!)",
            examples=[
                "Master shifu racoon wearing drip attire as a street gangster.",
                "A cute and adorable baby fox with big brown eyes, autumn leaves in the background enchanting,immortal,fluffy, shiny mane,Petals,fairyism,unreal engine 5 and Octane Render,highly detailed, photorealistic, cinematic, natural colors.",
                "The image features an intricately designed eye set against a circular backdrop adorned with ornate swirl patterns that evoke both realism and surrealism. At the center of attention is a strikingly vivid blue iris surrounded by delicate veins radiating outward from the pupil to create depth and intensity. The eyelashes are long and dark, casting subtle shadows on the skin around them which appears smooth yet slightly textured as if aged or weathered over time.\n\nAbove the eye, there's a stone-like structure resembling part of classical architecture, adding layers of mystery and timeless elegance to the composition. This architectural element contrasts sharply but harmoniously with the organic curves surrounding it. Below the eye lies another decorative motif reminiscent of baroque artistry, further enhancing the overall sense of eternity encapsulated within each meticulously crafted detail. \n\nOverall, the atmosphere exudes a mysterious aura intertwined seamlessly with elements suggesting timelessness, achieved through the juxtaposition of realistic textures and surreal artistic flourishes. Each component\u2014from the intricate designs framing the eye to the ancient-looking stone piece above\u2014contributes uniquely towards creating a visually captivating tableau imbued with enigmatic allure.",
            ],
            inputs=prompt_input,
        )

        understanding_button.click(
            multimodal_understanding, inputs=[image_input, question_input, und_seed_input, top_p, temperature], outputs=understanding_output
        )

        generation_button.click(fn=image_generation, inputs=[prompt_input, seed_input, cfg_weight_input, num_images], outputs=image_output)

    return demo
