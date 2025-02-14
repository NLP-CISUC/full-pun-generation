import polars as pl
import streamlit as st
from streamlit_sortables import sort_items
from utils import load_data

list_style = """
.sortable-item, .sortable-item:hover {
    background-color: var(--secondary-background-color);
    color: var(--foreground-color);
    border-color: var(--primary-color);
}
"""

df = load_data()


def format_headline(headline_id):
    return df.filter(pl.col("headline_id") == headline_id)["headline"].first()


def update_jokes():
    cur_headline = st.session_state["cur_headline"]
    jokes = (df.filter(pl.col("headline_id") == cur_headline)["generated"]
             .unique()
             .to_list())
    st.session_state["cur_jokes"] = jokes


st.set_page_config(layout="wide")
st.title("Avaliação")
st.selectbox("Notícia", df["headline_id"].unique(),
             format_func=format_headline, key="cur_headline",
             on_change=update_jokes)

col1, col2 = st.columns(2)

with col1:
    funniness = sort_items(st.session_state["cur_jokes"],
                           "Qual é a melhor piada?",
                           custom_style=list_style,
                           direction="vertical")
with col2:
    similarity = sort_items(st.session_state["cur_jokes"],
                            "Qual piada mais se relaciona com a notícia?",
                            custom_style=list_style,
                            direction="vertical")

funniness_df = (pl.DataFrame({"jokes": funniness})
                .with_row_index("funniness rank"))
similarity_df = (pl.DataFrame({"jokes": similarity})
                 .with_row_index("similarity rank"))
results_df = funniness_df.join(similarity_df, on="jokes")
st.dataframe(results_df)
