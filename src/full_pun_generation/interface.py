import os
from pathlib import Path

import gradio as gr
import polars as pl
import yaml

credentials_filepath = Path("data/config/credentials.yaml")
with credentials_filepath.open() as file:
    credentials = yaml.safe_load(file)


def get_current_pair(df, idx):
    if idx < df.height:
        return (df.item(idx, "generated"),
                df.item(idx, "generated_right"))
    return None, None


def load_data(request: gr.Request):
    filepath = credentials[request.username]["filepath"]
    df = pl.read_ndjson(filepath)
    return get_current_pair(df, 0)


with gr.Blocks() as demo:
    headline = gr.Markdown("## Headline")
    with gr.Row():
        left = gr.Markdown("Left")
        right = gr.Markdown("Right")

demo.launch(auth=[(c, credentials[c]["password"]) for c in credentials])
