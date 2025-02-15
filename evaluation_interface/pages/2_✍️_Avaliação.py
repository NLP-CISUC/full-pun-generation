import polars as pl
import streamlit as st
from streamlit_sortables import sort_items
from utils import get_results_path, load_config, load_custom_style, load_data

username = st.session_state.username
cfg = load_config()
splits = cfg["splits"]

results_path = get_results_path(username)
results_path.parent.mkdir(parents=True, exist_ok=True)

df = load_data().filter(pl.col("headline_id").is_in(splits[username]))


def format_headline():
    headline_id = st.session_state.cur_headline
    return df.filter(pl.col("headline_id") == headline_id)["headline"].first()


def update_jokes():
    cur_headline = st.session_state.cur_headline
    if ("results" in st.session_state and
        cur_headline in st.session_state.results["headline_id"]):
        cur_jokes = (st.session_state.results.filter(pl.col("headline_id") == cur_headline))
        st.session_state.cur_fun_rank = cur_jokes.sort("funniness rank")["generated"].to_list()
        st.session_state.cur_sim_rank = cur_jokes.sort("similarity rank")["generated"].to_list()
    else:
        jokes = (df.filter(pl.col("headline_id") == cur_headline)["generated"]
                 .unique()
                 .sample(fraction=1, shuffle=True)
                 .to_list())
        st.session_state.cur_fun_rank = jokes
        st.session_state.cur_sim_rank = jokes


def update_current_headline():
    st.session_state.cur_headline = splits[username][st.session_state.cur_idx]


def decrease_index():
    st.session_state.cur_idx -= 1
    update_current_headline()
    update_jokes()


def increase_index():
    st.session_state.cur_idx += 1
    update_current_headline()
    update_jokes()


#
# ------ Initialize state ------
#
if "results" not in st.session_state and results_path.exists():
    st.session_state.results = pl.read_ndjson(results_path)
if "cur_idx" not in st.session_state:
    st.session_state.cur_idx = 0
if "cur_headline" not in st.session_state:
    st.session_state.cur_headline = splits[username][st.session_state.cur_idx]
if "funiness_ranks" not in st.session_state or "similarity_ranks" not in st.session_state:
    update_jokes()


#
# ------ Streamlit page ------
#
with st.container(height=121, border=False):
    st.markdown(f"<p style=\"font-size:28px\">{format_headline()}</p>",
                unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large", border=True)
list_style = load_custom_style()
with col1:
    st.markdown("<p style=\"font-size:21px\">Qual piada <strong>tem mais piada</strong>?<p>",
                unsafe_allow_html=True)
    sort_items(st.session_state.cur_fun_rank, "",
               custom_style=list_style,
               direction="vertical",
               key=f"fun_rank_{st.session_state.cur_headline}")
with col2:
    st.markdown("<p style=\"font-size:21px\">Qual piada <strong>mais se relaciona</strong> com o título?<p>",
                unsafe_allow_html=True)
    sort_items(st.session_state.cur_sim_rank, "",
               custom_style=list_style,
               direction="vertical",
               key=f"sim_rank_{st.session_state.cur_headline}")

_, btn_col2, btn_col3, _ = st.columns([0.4, 0.1, 0.1, 0.4])
with btn_col2:
    first_example = (st.session_state.cur_idx == 0)
    st.button("Anterior", disabled=first_example, on_click=decrease_index)
with btn_col3:
    last_example = (st.session_state.cur_idx == len(splits[username]) - 1)
    st.button("Próximo", disabled=last_example, on_click=increase_index)

#
# ------ Consolidate and save results ------
#
if st.session_state[f"fun_rank_{st.session_state.cur_headline}"] is None:
    funniness = st.session_state.cur_fun_rank
else:
    funniness = st.session_state[f"fun_rank_{st.session_state.cur_headline}"][0]["items"]
if st.session_state[f"sim_rank_{st.session_state.cur_headline}"] is None:
    similarity = st.session_state.cur_sim_rank
else:
    similarity = st.session_state[f"sim_rank_{st.session_state.cur_headline}"][0]["items"]

funniness_df = (pl.DataFrame({"generated": funniness})
                .with_row_index("funniness rank"))
similarity_df = (pl.DataFrame({"generated": similarity})
                 .with_row_index("similarity rank"))
results_df = (funniness_df.join(similarity_df, on="generated")
              .join(df.filter(pl.col("headline_id") == st.session_state.cur_headline),
                    left_on="generated", right_on="generated")
              .select([pl.lit(username).alias("evaluator"),
                       "headline_id",
                       "headline",
                       "model",
                       "generated",
                       "pun sign",
                       "alternative sign",
                       "funniness rank",
                       "typicality",
                       "similarity rank",
                       "similarity"]))

if "results" in st.session_state:
    results_df = st.session_state.results.update(results_df,
                                                 on=["headline_id",
                                                     "generated"],
                                                 how="full")
st.session_state.results = results_df
st.session_state.results.write_ndjson(results_path)
