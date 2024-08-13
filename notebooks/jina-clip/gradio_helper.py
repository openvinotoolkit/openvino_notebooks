import gradio as gr
from PIL import Image

img_furseal = Image.open("./data/furseal.png")
img_coco = Image.open("./data/coco.jpg")


def make_demo(image_text_fn, text_text_fn, image_image_fn, model_choice_visible):
    with gr.Blocks() as demo:
        gr.Markdown("Discover simularity of text or image files using this demo.")
        quantized_model = gr.Checkbox(
            label="Use quantized int8 model", info="Model type. FP16 model is used by default.", visible=model_choice_visible, value=False
        )
        with gr.Tab("Text-Image"):
            with gr.Row():
                image_text_vis = gr.Image(label="Image", type="pil")
                text_text_vis = gr.Textbox(label="Labels", info="Use comma to separate sentences")
            text_image_button = gr.Button("Submit")
            with gr.Row():
                gr.Examples([img_furseal], image_text_vis)
                gr.Examples(["seal,rat,cobra"], text_text_vis)
            text_image_output = gr.Textbox(label="Results")
        with gr.Tab("Text-Text"):
            with gr.Row():
                text_text_1 = gr.Textbox(label="Text")
                text_text_2 = gr.Textbox(label="Text")
            text_text_button = gr.Button("Submit")
            with gr.Row():
                gr.Examples(["The breeding season for fur seals is from May to the end of November"], text_text_1)
                gr.Examples(["Fur seals feed on fish and squid"], text_text_2)
            text_text_output = gr.Textbox(label="Results")
        with gr.Tab("Image-Image"):
            with gr.Row():
                image_image_1 = gr.Image(label="Image", type="pil")
                image_image_2 = gr.Image(label="Image", type="pil")
            image_image_button = gr.Button("Submit")
            text_output = gr.Textbox(label="Results")
            with gr.Row():
                gr.Examples([img_furseal], image_image_1)
                gr.Examples([img_coco], image_image_2)
            image_image_output = gr.Textbox(label="Results")

        text_image_button.click(image_text_fn, inputs=[text_text_vis, image_text_vis, quantized_model], outputs=text_image_output)
        text_text_button.click(text_text_fn, inputs=[text_text_1, text_text_2, quantized_model], outputs=text_text_output)
        image_image_button.click(image_image_fn, inputs=[image_image_1, image_image_2, quantized_model], outputs=image_image_output)

    return demo
