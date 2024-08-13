from pathlib import Path
import gradio as gr
import requests
import tqdm

query = "Who developed the Theory of General Relativity?"
image_path = Path("example.jpg")

if not image_path.exists():
    r = requests.get("https://github.com/user-attachments/assets/a5bedef2-e915-4286-bcc9-d599083a99a6")
    with image_path.open("wb") as f:
        f.write(r.content)


def make_demo(build_saliency_map):
    def _process(image, query, n_iters, min_crop_size, _=gr.Progress(track_tqdm=True)):
        saliency_map = build_saliency_map(image, query, n_iters, min_crop_size, _tqdm=tqdm.tqdm, include_query=False)

        return saliency_map

    demo = gr.Interface(
        _process,
        [
            gr.Image(label="Image", type="pil"),
            gr.Textbox(label="Query"),
            gr.Slider(1, 10000, 2000, label="Number of iterations"),
            gr.Slider(1, 200, 50, label="Minimum crop size"),
        ],
        gr.Plot(label="Result"),
        examples=[[image_path, query]],
    )

    return demo
