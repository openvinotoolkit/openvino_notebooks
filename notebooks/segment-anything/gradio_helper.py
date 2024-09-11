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


def make_video_demo(segmenter, sample_path):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Video")
                coordinates = gr.Textbox(label="Coordinates")
                labels = gr.Textbox(label="Labels")
                submit_btn = gr.Button(value="Segment")
            with gr.Column():
                output_video = gr.Video(label="Output video")

        def on_video_change(video):
            segmenter.set_video(video)
            return video

        def segment_video(video, coordinates_txt, labels_txt):
            coordinates_np = []
            for coords in coordinates_txt.split(";"):
                temp = [float(numb) for numb in coords.split(",")]
                coordinates_np.append(temp)

            labels_np = []
            for l in labels_txt.split(","):
                labels_np.append(int(l))
            segmenter.set_video(video)
            segmenter.add_new_points_or_box(coordinates_np, labels_np)
            segmenter.propagate_in_video()
            video_out_path = segmenter.save_as_video()

            return video_out_path

        submit_btn.click(segment_video, inputs=[input_video, coordinates, labels], outputs=[output_video])
        input_video.upload(on_video_change, [input_video], [input_video])

        examples = gr.Examples(examples=[[sample_path / "coco.mp4", "430, 130; 500, 100", "1, 1"]], inputs=[input_video, coordinates, labels])

    return demo
