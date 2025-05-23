import gradio as gr
from surya.detection import batch_text_detection
from PIL import ImageDraw


def make_demo(model, processor, test_image):
    def predict(image):
        predictions = batch_text_detection([image], model, processor)

        image = image.copy()
        draw = ImageDraw.Draw(image)

        for polygon_box in predictions[0].bboxes:
            draw.rectangle(polygon_box.bbox, width=1, outline="red")

        return image

    demo = gr.Interface(
        predict,
        gr.Image(label="Image", type="pil", format="pil"),
        gr.Image(label="Result"),
        examples=[test_image],
    )

    return demo
