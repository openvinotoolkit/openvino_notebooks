from pathlib import Path
from typing import Callable
import gradio as gr
import pandas as pd
import requests

csv_file_name = "eu_city_population_top10.csv"

if not Path(csv_file_name).exists():
    r = requests.get("https://github.com/openvinotoolkit/openvino_notebooks/files/13215688/eu_city_population_top10.csv")
    with open(csv_file_name, "w") as f:
        f.write(r.text)


def display_table(csv_file_name):
    table = pd.read_csv(csv_file_name.name, delimiter=",")
    table = table.astype(str)
    return table


def make_demo(fn: Callable):
    with gr.Blocks(title="TAPAS Table Question Answering") as demo:
        with gr.Row():
            with gr.Column():
                search_query = gr.Textbox(label="Search query")
                csv_file = gr.File(label="CSV file")
                infer_button = gr.Button("Submit", variant="primary")
            with gr.Column():
                answer = gr.Textbox(label="Result")
                result_csv_file = gr.Dataframe(label="All data")

        examples = [
            [
                "What is the city with the highest population that is not a capital?",
                csv_file_name,
            ],
            ["In which country is Madrid?", csv_file_name],
            [
                "In which cities is the population greater than 2,000,000?",
                csv_file_name,
            ],
        ]
        gr.Examples(examples, inputs=[search_query, csv_file])

        # Callbacks
        csv_file.upload(display_table, inputs=csv_file, outputs=result_csv_file)
        csv_file.select(display_table, inputs=csv_file, outputs=result_csv_file)
        csv_file.change(display_table, inputs=csv_file, outputs=result_csv_file)
        infer_button.click(fn=fn, inputs=[search_query, csv_file], outputs=[answer, result_csv_file])
    return demo
