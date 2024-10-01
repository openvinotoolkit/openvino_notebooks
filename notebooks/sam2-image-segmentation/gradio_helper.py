import gradio as gr
import numpy as np
import cv2


def make_demo(segmenter):
    with gr.Blocks() as demo:
        with gr.Row():
            input_img = gr.Image(label="Input", type="numpy", height=480, width=480)
            output_img = gr.Image(label="Selected Segment", type="numpy", height=480, width=480)

        def on_image_change(img):
            segmenter.set_image(img)
            return img

        def get_select_coords(img, evt: gr.SelectData):
            pixels_in_queue = set()
            h, w = img.shape[:2]
            pixels_in_queue.add((evt.index[0], evt.index[1]))
            out = img.copy()
            while len(pixels_in_queue) > 0:
                pixels = list(pixels_in_queue)
                pixels_in_queue = set()
                color = np.random.randint(0, 255, size=(1, 1, 3))
                mask = segmenter.get_mask(pixels, img)
                mask_image = out.copy()
                mask_image[mask.squeeze(-1)] = color
                out = cv2.addWeighted(out.astype(np.float32), 0.7, mask_image.astype(np.float32), 0.3, 0.0)
            out = out.astype(np.uint8)
            return out

        input_img.select(get_select_coords, [input_img], output_img)
        input_img.upload(on_image_change, [input_img], [input_img])

    return demo
