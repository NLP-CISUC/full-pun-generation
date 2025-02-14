from pathlib import Path
import polars as pl
import streamlit as st
from streamlit_sortables import sort_items
from utils import load_data, custom_style

df = load_data()


def format_headline(headline_id):
    return df.filter(pl.col("headline_id") == headline_id)["headline"].first()


def update_jokes():
    cur_headline = st.session_state["cur_headline"]
    jokes = (df.filter(pl.col("headline_id") == cur_headline)["generated"]
             .unique()
             .to_list())
    st.session_state["cur_jokes"] = jokes


#
# ------ Streamlit page ------
#
st.selectbox("Notícia", df["headline_id"].unique(),
             format_func=format_headline, key="cur_headline",
             on_change=update_jokes)

if "cur_jokes" not in st.session_state:
    update_jokes()

col1, col2 = st.columns(2, gap="large", border=True)
list_style = custom_style()
with col1:
    st.markdown("Qual piada **tem mais piada**?")
    funniness = sort_items(st.session_state["cur_jokes"],
                           custom_style=list_style,
                           direction="vertical")
with col2:
    st.markdown("Qual piada **mais se relaciona** com a notícia?")
    similarity = sort_items(st.session_state["cur_jokes"], "",
                            custom_style=list_style,
                            direction="vertical")

#
# ------ Consolidate results ------
#
username = st.session_state["username"]
funniness_df = (pl.DataFrame({"joke": funniness})
                .with_row_index("funniness rank"))
similarity_df = (pl.DataFrame({"joke": similarity})
                 .with_row_index("similarity rank"))
results_df = (funniness_df.join(similarity_df, on="joke")
              .join(df.filter(pl.col("headline_id") == st.session_state["cur_headline"]),
                    left_on="joke", right_on="generated")
              .select([pl.lit(username).alias("evaluator"),
                       "headline_id",
                       "headline",
                       "model",
                       "joke",
                       "pun sign",
                       "alternative sign",
                       "funniness rank",
                       "typicality",
                       "similarity rank",
                       "similarity"]))

#
# ------ Save results ------
#
if "results" in st.session_state:
    results_df = st.session_state["results"].update(results_df,
                                                    on=["headline_id", "joke"],
                                                    how="full")
st.session_state["results"] = results_df

results_path = Path("results/evaluation") / f"{username}.jsonl"
results_path.parent.mkdir(parents=True, exist_ok=True)
st.session_state["results"].write_ndjson(results_path)
