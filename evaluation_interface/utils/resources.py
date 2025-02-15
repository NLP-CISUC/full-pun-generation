from pathlib import Path

import polars as pl
import streamlit as st
import yaml

INTERFACE_ROOT = Path(__file__).parent.parent.resolve()

@st.cache_data
def load_config():
    config_path = INTERFACE_ROOT / "config" / "config.yaml"
    return yaml.safe_load(config_path.read_text())

@st.cache_data
def load_data():
    cfg = load_config()
    data_path = INTERFACE_ROOT / cfg["paths"]["data"]
    return pl.read_ndjson(data_path)


@st.cache_data
def load_custom_style():
    cfg = load_config()
    style_path = INTERFACE_ROOT / cfg["paths"]["style"]
    return style_path.read_text()

def get_results_path(username):
    cfg = load_config()
    results_path = INTERFACE_ROOT / cfg["paths"]["results"] / f"{username}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    return results_path
