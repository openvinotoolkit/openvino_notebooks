import gradio as gr
import numpy as np
import cv2


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
                if len(temp) == 4:
                    box_coords = np.array(temp).reshape(2, 2)
                    coordinates_np.append(box_coords)
                else:
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

        examples = gr.Examples(
            examples=[[sample_path / "coco.mp4", "430, 130; 500, 100", "1, 1"], [sample_path / "coco.mp4", "380, 75, 530, 260", "2, 3"]],
            inputs=[input_video, coordinates, labels],
        )

    return demo
