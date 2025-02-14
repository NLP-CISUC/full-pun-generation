import streamlit as st
from .process_generations import load_data

@st.cache_data
def custom_style():
    with open("config/style.css") as f:
        return f.read()
